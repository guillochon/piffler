"""Recovery rounds: redo chosen pages of a finished book, without reprocessing it.

Every page remembers what it was made from (its hold of the video, the frames
used, and where the book and gutter were thought to be at that point), so a
page can be made again on its own with different settings. Settings live in
``overrides.json`` in the output folder, keyed by page id (``s142-left``), and
every later run, full or fix, applies them.

    piffler fix OUT --pages 68,88 --set gutter=0.53      # by hand
    piffler fix OUT --drop 143                           # remove a page
    piffler fix OUT --keep s150-left                     # restore a dropped one
    piffler fix OUT --auto                               # try variants on flagged pages

Only the frames the chosen holds need are decoded, and only the redone pages'
images are made again; the rest are reused under their new numbers.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .export import write_outputs
from .ocr import OCREngine, Word
from .pipeline import (PageResult, _load_stats, _Orientation, _PagePrior, _process_segment, _resize_long,
                       all_hold_frames, hold_overrides, load_overrides, write_pages)
from .segment import Segment, find_segments
from .video import VideoSource

log = logging.getLogger(__name__)

SETTINGS = {"ref_frame", "exclude_frames", "frames", "fill_fingers", "misfit", "expand_page", "outline",
            "layout", "gutter", "dewarp", "deshear", "drop", "keep", "note", "score"}


class Book:
    """A finished output folder, loaded back as pages."""

    def __init__(self, out_dir: str, video: Optional[str] = None):
        self.out_dir = out_dir
        with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)
        c = dict(self.manifest["config"])
        self.cfg = Config(**{k: v for k, v in c.items() if k in Config.__dataclass_fields__})
        self.video = video or self.manifest["video"]
        self.fps = float(self.manifest["fps"])
        self.pages: List[PageResult] = []
        missing = 0
        for d in self.manifest["pages"]:
            p = PageResult(d["page"], d["segment"], d["frame_index"], d["time_s"],
                           image_path=os.path.join(out_dir, d["image"]) if d.get("image") else "",
                           flags=list(d["flags"]), info=dict(d["info"]))
            p.photo_path = os.path.join(out_dir, d["photo"]) if d.get("photo") else ""
            stem = os.path.join(out_dir, "pages", f"page_{d['page']:04d}")
            with open(stem + ".txt", encoding="utf-8") as f:
                p.text = f.read()
            try:
                with open(stem + ".json", encoding="utf-8") as f:
                    side = json.load(f)
                p.words = [Word(t, float(sc), np.float32(b)) for t, sc, b in side["words"]]
                p.image_scale = side.get("image_scale", 1.0)
                p.ctx = side.get("ctx")
                p.size = side.get("size")
            except (OSError, ValueError, KeyError):
                missing += 1
                p.ctx = None
            p._files = {"photo": p.photo_path, "image": p.image_path}
            self.pages.append(p)
        if missing:
            log.warning("%d pages have no saved context (made by an older piffler); run the book once more "
                        "to make them fixable", missing)
        self.overrides = load_overrides(out_dir)
        st = _load_stats(os.path.join(out_dir, "analysis_cache.npz"), self._cache_key())
        if st is None:
            raise RuntimeError("analysis_cache.npz is missing or from another video; run the book once first")
        self.segs = find_segments(st, self.cfg, self.fps)

    def _cache_key(self) -> str:
        z = np.load(os.path.join(self.out_dir, "analysis_cache.npz"), allow_pickle=False)
        return str(z["key"])

    # -- addressing -----------------------------------------------------------
    def resolve(self, refs: Iterable) -> List[str]:
        """Page numbers or ids -> ids (dropped pages are addressed by id)."""
        by_no = {p.page_no: p.id for p in self.pages}
        known = {p.id for p in self.pages} | {d.get("id") for d in self.manifest.get("dropped", [])}
        out = []
        for r in refs:
            r = str(r).strip()
            if not r:
                continue
            if r.isdigit():
                if int(r) not in by_no:
                    raise ValueError(f"no page {r}")
                out.append(by_no[int(r)])
            elif r in known:
                out.append(r)
            else:
                raise ValueError(f"no page {r!r} (use a page number or an id like s142-left)")
        return out

    def save_overrides(self):
        path = os.path.join(self.out_dir, "overrides.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(self.overrides.items(), key=lambda kv: _id_key(kv[0]))), f, indent=2)

    def context(self, si: int) -> Optional[Dict]:
        for p in self.pages:
            if p.segment == si and p.ctx:
                return p.ctx
        # a dropped page's hold: borrow the nearest page's context
        near = [p for p in self.pages if p.ctx]
        if not near:
            return None
        return min(near, key=lambda p: abs(p.segment - si)).ctx

    # -- redo -----------------------------------------------------------------
    def redo(self, plans: Dict[int, List[Dict]], progress: Callable[[str, float], None] = None
             ) -> Dict[int, List[List[PageResult]]]:
        """For each hold, process it once per settings dict in its plan.
        Returns {hold: [results for each plan entry]}. Frames are decoded in
        one pass and each hold is done as soon as its frames are in."""
        say = progress or (lambda stage, frac: None)
        need: Dict[int, List[int]] = {}
        for si, variants in plans.items():
            seg = self.segs[si]
            ids = set(seg.candidates)
            if any(v.get("frames") == "all" for v in variants):
                ids |= set(all_hold_frames(seg))
            ctx = self.context(si) or {}
            ids |= set((ctx.get("extra") or {}).get("ids", []))
            for v in variants:
                if v.get("ref_frame") is not None:
                    ids.add(int(v["ref_frame"]))
            need[si] = sorted(ids)
        users: Dict[int, int] = {}
        for ids in need.values():
            for i in ids:
                users[i] = users.get(i, 0) + 1
        last = {si: max(ids) for si, ids in need.items()}
        engine = OCREngine(self.cfg.lang_det_long_side, self.cfg.ocr_min_score, self.cfg.gpu) if self.cfg.ocr else None
        rots = [p.ctx.get("rot") for p in self.pages if p.ctx and p.ctx.get("rot") is not None]
        locked = max(set(rots), key=rots.count) if rots else 0
        out: Dict[int, List[List[PageResult]]] = {}
        buf: Dict[int, np.ndarray] = {}
        pending = sorted(need, key=lambda si: last[si])
        done = 0
        with VideoSource(self.video, rotate=self.cfg.rotate, max_bad_reads=self.cfg.max_bad_reads,
                         use_gpu=self.cfg.gpu) as src:
            for idx, frame in src.iter_fetch(users):
                buf[idx] = _resize_long(frame, self.cfg.max_frame_side)
                while pending and last[pending[0]] <= idx:
                    si = pending.pop(0)
                    out[si] = [self._redo_one(si, v, buf, engine, locked) for v in plans[si]]
                    for i in need[si]:
                        users[i] -= 1
                        if users[i] == 0:
                            buf.pop(i, None)
                    done += 1
                    say("redo", done / len(need))
        for si in pending:  # frames that failed to decode
            out[si] = [self._redo_one(si, v, buf, engine, locked) for v in plans[si]]
        return out

    def _redo_one(self, si: int, ov: Dict, buf: Dict[int, np.ndarray], engine, locked: int) -> List[PageResult]:
        seg0 = self.segs[si]
        cands = all_hold_frames(seg0) if ov.get("frames") == "all" else list(seg0.candidates)
        seg = Segment(seg0.start, seg0.end, seg0.t0, seg0.t1, cands,
                      list(seg0.scores) + [min(seg0.scores)] * (len(cands) - len(seg0.scores)), list(seg0.flags))
        frames = {i: buf[i] for i in cands if i in buf}
        ctx = self.context(si) or {}
        prior = _PagePrior()
        prior.quads = [np.float32(q) for q in ctx.get("prior_quads", [])]
        prior.gutters = list(ctx.get("prior_gutters", []))
        orient = _Orientation(engine)
        orient.locked = ctx.get("rot") if ctx.get("rot") is not None else locked
        ex = ctx.get("extra")
        extra = None
        if ex and ov.get("frames") != "own":
            eframes = {i: buf[i] for i in ex["ids"] if i in buf}
            extra = (ex["start"], ex["ids"], eframes) if eframes else None
        try:
            return _process_segment(si, seg, frames, self.cfg, engine, orient, self.fps,
                                    os.path.join(self.out_dir, "pages"), None, prior, False, extra, ov=ov)
        except Exception as e:
            log.warning("redoing hold %d with %s failed: %s", si, ov, e)
            return []

    # -- reassemble -------------------------------------------------------------
    def apply(self, fresh: Dict[int, List[PageResult]], note: str = "") -> dict:
        """Splice redone holds into the book, apply drop/keep overrides, rewrite outputs."""
        pages = list(self.pages)
        for si, rs in fresh.items():
            pos = [k for k, p in enumerate(pages) if p.segment == si]
            for k in reversed(pos):
                pages.pop(k)
            at = pos[0] if pos else next((k for k, p in enumerate(pages) if p.frame_index > rs[0].frame_index),
                                         len(pages)) if rs else len(pages)
            pages[at:at] = rs
        dropped = [d for d in self.manifest.get("dropped", []) if not self.overrides.get(d.get("id"), {}).get("keep")]
        for p in [p for p in pages if self.overrides.get(p.id, {}).get("drop")]:
            pages.remove(p)
            dropped.append({"id": p.id, "segment": p.segment, "frame_index": p.frame_index,
                            "time_s": round(p.time_s, 2), "half": p.info.get("half"), "chars": len(p.text),
                            "reason": "dropped by override"})
        write_pages(pages, os.path.join(self.out_dir, "pages"), self.cfg)
        meta = {k: v for k, v in self.manifest.items() if k not in ("pages", "needs_review", "pdf")}
        meta["dropped"] = dropped
        meta.setdefault("fixes", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note,
                                             "redone": sorted(f"s{si}" for si in fresh)})
        self.manifest = write_outputs(self.out_dir, pages, meta, self.cfg.pdf)
        self.pages = pages
        return self.manifest


def _id_key(pid: str):
    m = re.match(r"s(\d+)(?:-(\w+))?", pid)
    return (int(m.group(1)), m.group(2) or "") if m else (10 ** 9, pid)


def _seg_of(pid: str) -> int:
    return int(re.match(r"s(\d+)", pid).group(1))


# ------------------------------------------------------------------ scoring

def _bigrams(text: str):
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    w = re.findall(r"[^\W\d_]{3,}", t)
    return set(zip(w, w[1:]))


def edge_words(p: PageResult) -> int:
    """Words touching the page's left or right edge: text cut off by the outline or the split."""
    img = getattr(p, "_image", None)
    if img is not None:
        pw = img.shape[1]
    elif getattr(p, "size", None):
        pw = p.size[1]
    else:
        im = cv2.imread(p.photo_path) if p.photo_path else None
        if im is None:
            return 0
        pw = im.shape[1]
    e = 0.008 * pw
    return sum(1 for w in p.words if w.x0 <= e or w.box[:, 0].max() >= pw - e)


def score(rs: List[PageResult], neighbours: List[PageResult]) -> float:
    """How good a redone hold's pages are: confidently read text, minus text a
    hand may hide, words cut at an edge, and text that belongs to a neighbour."""
    near = set()
    for q in neighbours:
        near |= _bigrams(q.text)
    total = 0.0
    for r in rs:
        conf = sum(w.score for w in r.words if w.score >= 0.6)
        hidden = float(r.info.get("hidden_text_frac", 0.0) or 0.0)
        s = conf * (1.0 - min(0.5, 25.0 * hidden)) - 4.0 * edge_words(r)
        if "finger_may_cover_text" in r.flags:
            s -= 8.0
        b = _bigrams(r.text)
        if len(b) >= 20 and near and len(b & near) / len(b) > 0.4:  # it's a neighbour's text
            s -= 0.5 * conf
        total += s
    return total


def problems(p: PageResult) -> List[str]:
    f = set(p.flags)
    out = []
    if "finger_may_cover_text" in f:
        out.append("hidden_text")
    if edge_words(p) >= 2:
        out.append("text_at_edge")
    if "low_ocr_confidence" in f and len(p.text) > 100:
        out.append("ghosting")
    elif p.words and len(p.text) > 200 and np.mean([w.score for w in p.words]) < 0.9:
        out.append("ghosting")
    if "page_from_neighbors" in f:
        out.append("outline_guessed")
    if "possible_duplicate" in f:
        out.append("overlaid_pages")
    return out


def variants(probs: List[str], ctx: Optional[Dict]) -> List[Dict]:
    """Settings worth trying for these problems (on top of the page's current ones)."""
    out: List[Dict] = []
    split = (ctx or {}).get("split")
    for pr in probs:
        if pr == "hidden_text":
            out += [{"frames": "all"}, {"frames": "all", "misfit": 0.05}]
        elif pr == "text_at_edge":
            out += [{"expand_page": 0.03}, {"outline": "neighbors"}]
            if split is not None:
                out += [{"gutter": round(split + d, 3)} for d in (0.02, -0.02, 0.04, -0.04)]
        elif pr == "ghosting":
            out += [{"misfit": 0.04}, {"fill_fingers": False}, {"frames": "all", "misfit": 0.04}]
        elif pr == "outline_guessed":
            out += [{"frames": "all"}, {"expand_page": 0.03}]
    seen, uniq = set(), []
    for v in out:
        k = json.dumps(v, sort_keys=True)
        if k not in seen:
            seen.add(k)
            uniq.append(v)
    return uniq


# ------------------------------------------------------------------- entry points

def fix(out_dir: str, pages: Iterable = (), settings: Optional[Dict] = None, drop: Iterable = (),
        keep: Iterable = (), auto: bool = False, video: Optional[str] = None,
        progress: Callable[[str, float], None] = None) -> dict:
    """Apply recovery settings to a finished book. Returns a report dict."""
    book = Book(out_dir, video)
    report: Dict = {"set": {}, "dropped": [], "restored": [], "auto": []}
    targets = book.resolve(pages)
    for pid in book.resolve(drop):
        book.overrides.setdefault(pid, {})["drop"] = True
        report["dropped"].append(pid)
    for pid in book.resolve(keep):
        o = book.overrides.setdefault(pid, {})
        o.pop("drop", None)
        o["keep"] = True
        report["restored"].append(pid)
    holds = set()
    if settings:
        unknown = set(settings) - SETTINGS
        if unknown:
            raise ValueError(f"unknown setting(s): {', '.join(sorted(unknown))}")
        for pid in targets:
            book.overrides.setdefault(f"s{_seg_of(pid)}", {}).update(settings)
            report["set"][pid] = settings
            holds.add(_seg_of(pid))
    holds |= {_seg_of(pid) for pid in report["restored"]}
    plans = {si: [hold_overrides(book.overrides, si)] for si in holds}

    if auto:
        chosen = targets or [p.id for p in book.pages if p.needs_review or "text_at_edge" in problems(p)]
        report["checked"] = len(chosen)
        by_hold: Dict[int, List[str]] = {}
        for pid in chosen:
            p = next((q for q in book.pages if q.id == pid), None)
            if p is None or not p.ctx:
                continue
            probs = problems(p)
            if probs and p.segment not in holds:
                by_hold.setdefault(p.segment, []).extend(probs)
        for si, probs in by_hold.items():
            base = hold_overrides(book.overrides, si)
            vs = [dict(base, **v) for v in variants(sorted(set(probs)), book.context(si))]
            if vs:
                plans[si] = [base] + vs
    if not plans:
        book.save_overrides()
        if report["dropped"] or report["restored"]:
            book.apply({}, note="drop/keep")
            report["pages"] = len(book.pages)
            report["needs_review"] = book.manifest.get("needs_review", [])
        return report

    results = book.redo(plans, progress)
    # a redone hold yields both halves again: pages dropped earlier stay dropped
    gone = {d.get("id") for d in book.manifest.get("dropped", [])
            if not book.overrides.get(d.get("id"), {}).get("keep")}
    results = {si: [[r for r in rs if r.id not in gone] for rs in runs] for si, runs in results.items()}
    fresh: Dict[int, List[PageResult]] = {}
    for si, runs in results.items():
        if len(plans[si]) == 1 or not auto or si in holds:
            if runs and runs[0]:
                fresh[si] = runs[0]
            continue
        pos = [k for k, p in enumerate(book.pages) if p.segment == si]
        nb = [book.pages[k] for k in range(max(0, min(pos) - 2), min(len(book.pages), max(pos) + 3))
              if book.pages[k].segment != si] if pos else []
        scored = [(score(rs, nb), k) for k, rs in enumerate(runs) if rs]
        if not scored:
            continue
        base_s = next((s for s, k in scored if k == 0), None)
        best_s, best_k = max(scored)
        entry = {"hold": f"s{si}", "problems": sorted(set(by_hold.get(si, []))), "baseline": round(base_s or 0, 1),
                 "tried": len(runs) - 1, "best": round(best_s, 1)}
        if best_k != 0 and base_s is not None and best_s >= base_s + max(3.0, 0.02 * abs(base_s)):
            chosen_ov = plans[si][best_k]
            new = {k: v for k, v in chosen_ov.items() if plans[si][0].get(k) != v}
            book.overrides.setdefault(f"s{si}", {}).update(new)
            book.overrides[f"s{si}"]["note"] = "auto: " + ", ".join(entry["problems"])
            book.overrides[f"s{si}"]["score"] = [entry["baseline"], entry["best"]]
            fresh[si] = runs[best_k]
            entry["applied"] = new
        else:
            entry["applied"] = None  # nothing beat the page as it is
        report["auto"].append(entry)
    book.save_overrides()
    note = "auto" if auto else "manual"
    book.apply(fresh, note=note)
    report["pages"] = len(book.pages)
    report["needs_review"] = book.manifest.get("needs_review", [])
    return report
