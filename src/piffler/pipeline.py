"""End-to-end: video -> still segments -> clean pages -> OCR -> outputs."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .config import Config
from .dewarp import dewarp, flatten_illumination, text_mask
from .export import write_outputs
from .hands import remove_fingers
from .ocr import OCREngine, Word, best_rotation, rotate, words_to_text
from .page import detect_page, rectify, split_pages
from .segment import FrameStats, Segment, analyze, find_segments
from .video import VideoSource

log = logging.getLogger(__name__)

REVIEW_FLAGS = {"finger_may_cover_text", "low_quality_frame", "short_hold", "page_not_found",
                "processing_error", "gutter_guessed", "no_text_found", "no_still_segments",
                "possible_duplicate", "low_ocr_confidence"}


@dataclass
class PageResult:
    page_no: int
    segment: int
    frame_index: int
    time_s: float
    image_path: str = ""
    text: str = ""
    words: List[Word] = field(default_factory=list, repr=False)
    flags: List[str] = field(default_factory=list)
    info: Dict = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        return float(np.mean([w.score for w in self.words])) if self.words else 0.0

    @property
    def needs_review(self) -> bool:
        return bool(REVIEW_FLAGS.intersection(self.flags))

    def to_dict(self, root: str = ""):
        return {
            "page": self.page_no, "segment": self.segment, "frame_index": self.frame_index,
            "time_s": round(self.time_s, 2),
            "image": os.path.relpath(self.image_path, root) if root and self.image_path else self.image_path,
            "chars": len(self.text), "ocr_mean_score": round(self.mean_score, 3),
            "flags": sorted(set(self.flags)), "needs_review": self.needs_review, "info": self.info,
        }


def _resize_long(img, long_side):
    s = long_side / float(max(img.shape[:2]))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def page_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of ORB features consistent with one homography (0..1)."""
    ga = cv2.cvtColor(_resize_long(a, 700), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(_resize_long(b, 700), cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1500)
    k1, d1 = orb.detectAndCompute(ga, None)
    k2, d2 = orb.detectAndCompute(gb, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        # featureless (blank page / cover): fall back to image correlation
        x = cv2.resize(ga, (64, 64)).astype(np.float32); y = cv2.resize(gb, (64, 64)).astype(np.float32)
        x -= x.mean(); y -= y.mean()
        return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-6)) * 0.5
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 12:
        return 0.0
    src = np.float32([k1[m.queryIdx].pt for m in good]); dst = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or inl is None:
        return 0.0
    return float(inl.sum()) / min(len(k1), len(k2))


class _Orientation:
    """Decide page rotation once per video from the first pages with text."""

    def __init__(self, engine: Optional[OCREngine]):
        self.engine, self.votes, self.locked = engine, [], None

    def get(self, img) -> int:
        if self.locked is not None or self.engine is None:
            return self.locked or 0
        r = best_rotation(img, self.engine)
        self.votes.append(r)
        if len(self.votes) >= 2 and self.votes[-1] == self.votes[-2]:
            self.locked = r
        elif len(self.votes) >= 4:
            self.locked = max(set(self.votes), key=self.votes.count)
        return r


class _PagePrior:
    """Where the book has been in this video, from pages found by their text.

    Phone-on-a-stand footage keeps the book roughly in place, so a page with
    little text (a photo, a chapter opener) borrows the location of its
    neighbours instead of latching onto a white pan or countertop.
    """

    def __init__(self):
        self.quads: List[np.ndarray] = []
        self.gutters: List[float] = []

    def add(self, quad: np.ndarray):
        self.quads = (self.quads + [quad])[-5:]

    def add_gutter(self, rel: float):
        g = self.gutter()
        if g is None or abs(rel - g) < 0.06:  # don't let one odd spread move it
            self.gutters = (self.gutters + [rel])[-9:]

    def gutter(self) -> Optional[float]:
        return float(np.median(self.gutters)) if self.gutters else None

    def get(self, shape):
        if not self.quads:
            return None
        quad = np.median(np.stack(self.quads), axis=0).astype(np.float32)
        mask = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
        return quad, mask


def _page_thumb(ref: np.ndarray, cfg: Config, prior: "_PagePrior") -> Optional[np.ndarray]:
    """The page alone, rectified and small, for duplicate checks.

    Whole frames won't do: with a fixed camera the background dominates and
    every frame of the video looks like every other.
    """
    det = _resize_long(ref, cfg.detect_long_side)
    known = prior.get(det.shape)
    quad, _, flags = detect_page(det, known[1] if known is not None else None)
    if "page_by_brightness" in flags:
        if known is None:
            return None  # location is a guess: comparing guesses would match anything
        quad = known[0]
    return _resize_long(rectify(det, quad), 700)


class _Deferred(Exception):
    """Page can't be located yet; retry once neighbouring pages give a prior."""


def _process_segment(si: int, seg: Segment, frames: Dict[int, np.ndarray], cfg: Config,
                     engine: Optional[OCREngine], orient: _Orientation, fps: float,
                     out_pages: str, debug_dir: Optional[str], prior: _PagePrior,
                     allow_defer: bool) -> List[PageResult]:
    cands = [i for i in seg.candidates if i in frames]
    if not cands:
        r = PageResult(0, si, seg.best, seg.t0, flags=seg.flags + ["frame_decode_failed"])
        return [r]
    ref_idx = cands[0]  # candidates are sorted best-first
    ref = frames[ref_idx]
    flags = list(seg.flags)
    info: Dict = {"hold_s": round(seg.t1 - seg.t0, 2)}

    # find the page on a reduced copy, then crop full-resolution frames to it, so
    # a book filling a small part of a 4K frame keeps all its pixels for OCR
    det = _resize_long(ref, cfg.detect_long_side)
    ds = ref.shape[1] / float(det.shape[1])
    known = prior.get(det.shape)
    quad, pmask, pflags = detect_page(det, known[1] if known is not None else None)
    if "page_by_brightness" in pflags:
        if known is None and allow_defer:
            raise _Deferred()
        if known is not None:
            quad, pmask = known
            pflags = ["page_from_neighbors"]
    elif "page_near_neighbors" in pflags:
        # little print to anchor on, so a hand or a shaded page edge can drag a
        # corner off; trust the neighbours' corner wherever the two disagree
        pq = known[0]
        tol = 0.04 * float(np.hypot(*(pq.max(0) - pq.min(0))))
        far = np.hypot(*(quad - pq).T) > tol
        quad = np.where(far[:, None], pq, quad).astype(np.float32)
        if far.any():
            pmask = np.zeros(det.shape[:2], np.uint8)
            cv2.fillPoly(pmask, [quad.astype(np.int32)], 255)
    else:
        prior.add(quad)
    flags += pflags
    quad = quad * ds
    H, W = ref.shape[:2]
    x0, y0 = quad.min(0)
    x1, y1 = quad.max(0)
    pad = 0.04 * max(x1 - x0, y1 - y0)
    x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    x1, y1 = int(min(W, x1 + pad + 1)), int(min(H, y1 + pad + 1))
    cs = min(1.0, cfg.work_long_side / float(max(x1 - x0, y1 - y0)))

    def crop(f):
        f = f[y0:y1, x0:x1]
        return cv2.resize(f, None, fx=cs, fy=cs, interpolation=cv2.INTER_AREA) if cs < 1 else f

    frames = {i: crop(frames[i]) for i in cands}
    ref = frames[ref_idx]
    quad = ((quad - [x0, y0]) * cs).astype(np.float32)
    pmask = cv2.resize(cv2.resize(pmask, (W, H), interpolation=cv2.INTER_NEAREST)[y0:y1, x0:x1],
                       (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)

    clean = ref
    if cfg.remove_fingers:
        others = [frames[i] for i in cands[1:]]
        clean, residual, fstats = remove_fingers(ref, others, pmask, cfg.skin_paper_delta)
        info["fingers"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in fstats.items()}
        if residual.any():
            g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
            tm = text_mask(g)
            near = cv2.dilate(residual, np.ones((25, 25), np.uint8))
            covered = float(((tm > 0) & (near > 0) & (pmask > 0)).sum())
            if covered > 0.0005 * pmask.astype(bool).sum() and fstats["residual_frac"] > 0.002:
                flags.append("finger_may_cover_text")
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"seg{si:04d}_ref.jpg"), _resize_long(ref, 1200))
            cv2.imwrite(os.path.join(debug_dir, f"seg{si:04d}_clean.jpg"), _resize_long(clean, 1200))

    region = rectify(clean, quad)
    rot = 0 if cfg.rotate else orient.get(region)  # forced rotation is applied at decode
    if rot:
        region = rotate(region, rot)
        info["auto_rotated"] = rot
    aspect = region.shape[1] / float(region.shape[0])
    hint = prior.gutter() if aspect > 0.85 and cfg.layout == "auto" else None
    log.debug("segment %d: splitting (aspect %.2f)", si, aspect)
    halves, sflags, sure = split_pages(region, cfg.layout, hint)
    if sure is not None:
        prior.add_gutter(sure)
    elif hint is None and aspect > 0.85 and cfg.layout == "auto" and allow_defer:
        raise _Deferred()  # a spread-shaped page with an unclear gutter: wait for a confident one
    flags += sflags

    results = []
    for hi, page in enumerate(halves):
        pf = list(flags)
        pinfo = dict(info)
        if cfg.dewarp:
            try:
                page, dinfo = dewarp(page)
                pinfo["dewarp"] = dinfo
            except Exception as e:
                log.warning("dewarp failed on segment %d: %s", si, e)
                pf.append("dewarp_failed")
        page = flatten_illumination(page)
        words: List[Word] = []
        if engine is not None:
            words = engine(page)
            if len(halves) > 1:
                # each half overlaps the facing page a little; words lying wholly in
                # that strip are the facing page's line starts or ends
                strip, pw = 2 * int(0.015 * region.shape[1]), page.shape[1]
                words = [w for w in words if (w.x0 <= pw - strip if hi == 0 else w.box[:, 0].max() >= strip)]
            if not words:
                pf.append("no_text_found")
            elif np.mean([w.score for w in words]) < 0.75:
                pf.append("low_ocr_confidence")
        r = PageResult(0, si, ref_idx, ref_idx / fps, words=words, flags=pf, info=pinfo)
        r.text = words_to_text(words)
        r._image = page
        if len(halves) > 1:
            r.info["half"] = "left" if hi == 0 else "right"
        results.append(r)
    return results


def _load_stats(path: str, key: str) -> Optional[FrameStats]:
    """Frame analysis from an earlier run on the same video and settings."""
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["key"]) != key:
                return None
            return FrameStats(**{f: z[f] for f in FrameStats.__dataclass_fields__})
    except Exception:
        return None


def _save_stats(path: str, key: str, stats: FrameStats):
    try:
        np.savez(path, key=np.array(key), **{f: getattr(stats, f) for f in FrameStats.__dataclass_fields__})
    except Exception as e:  # a cache is never worth failing a run over
        log.warning("could not save analysis cache: %s", e)


def process_video(video_path: str, out_dir: str, cfg: Optional[Config] = None,
                  progress: Optional[Callable[[str, float], None]] = None) -> List[PageResult]:
    cfg = cfg or Config()
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    out_pages = os.path.join(out_dir, "pages")
    os.makedirs(out_pages, exist_ok=True)
    debug_dir = os.path.join(out_dir, "debug") if cfg.save_debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    say = progress or (lambda stage, frac: None)
    st = os.stat(video_path) if os.path.exists(video_path) else None
    cache_path = os.path.join(out_dir, "analysis_cache.npz")
    cache_key = repr((os.path.abspath(video_path), st and st.st_size, st and int(st.st_mtime),
                      cfg.analysis_fps, cfg.analysis_width, cfg.rotate))

    with VideoSource(video_path, rotate=cfg.rotate, max_bad_reads=cfg.max_bad_reads) as src:
        fps = src.info.fps
        log.info("Video: %dx%d @ %.2f fps", src.info.width, src.info.height, fps)
        say("analyzing", 0.0)
        hint = max(1, src.info.frame_count_hint)
        stats = _load_stats(cache_path, cache_key)
        if stats is None:
            stats = analyze(src, cfg, progress=lambda i: say("analyzing", min(1.0, i / hint)))
            _save_stats(cache_path, cache_key, stats)
        else:
            log.info("Reusing frame analysis from %s", cache_path)
        segs = find_segments(stats, cfg, fps)
        log.info("Found %d still segments", len(segs))
        if debug_dir:
            np.savetxt(os.path.join(debug_dir, "frame_stats.csv"),
                       np.stack([stats.idx, stats.t, stats.motion, stats.sharp, stats.bright, stats.skin], 1),
                       delimiter=",", header="frame,t,motion,sharpness,brightness,skin", fmt="%.4f")

        engine = OCREngine(cfg.lang_det_long_side, cfg.ocr_min_score) if cfg.ocr else None
        orient = _Orientation(engine)
        # map frame -> segment, and the last frame each segment needs
        owner, last_needed = {}, {}
        for si, s in enumerate(segs):
            for i in s.candidates:
                owner.setdefault(i, []).append(si)
            last_needed[si] = max(s.candidates)
        buf: Dict[int, Dict[int, np.ndarray]] = {}
        results: List[PageResult] = []
        kept_refs: List = []  # (segment index, ref thumbnail, score, [result idx])
        done = set()
        prior = _PagePrior()
        deferred: Dict[int, Dict[int, np.ndarray]] = {}

        def finish(si, allow_defer=True):
            seg = segs[si]
            frames = buf.pop(si, None) or deferred.pop(si, None) or {}
            say("pages", (si + 1) / max(1, len(segs)))
            # duplicate check against recent pages (user hesitated, or flipped back)
            thumb, replaces = None, None
            if frames:
                ref = frames.get(seg.candidates[0]) if seg.candidates[0] in frames else next(iter(frames.values()))
                thumb = _page_thumb(ref, cfg, prior)
            if cfg.dedupe and thumb is not None and kept_refs:
                # nearest kept pages in video time (deferral can process out of order)
                before = sorted((r for r in kept_refs if r[0] < si), key=lambda r: -r[0])[:3]
                after = sorted((r for r in kept_refs if r[0] > si), key=lambda r: r[0])[:1]
                for entry in after + before:
                    psi, pthumb, pscore, pidx = entry
                    sim = page_similarity(thumb, pthumb)
                    log.debug("segment %d vs %d: sim %.3f", si, psi, sim)
                    if sim < cfg.dedupe_similarity:
                        continue
                    adjacent = any(entry is e for e in after) or (before and entry is before[0])
                    if adjacent:  # same page held twice in a row: keep the better capture
                        if max(seg.scores) <= pscore:
                            log.info("segment %d duplicates %d (sim %.2f); skipped", si, psi, sim)
                            return
                        replaces = entry
                    else:
                        seg.flags.append("possible_duplicate")
                    break
            try:
                rs = _process_segment(si, seg, frames, cfg, engine, orient, fps, out_pages, debug_dir,
                                      prior, allow_defer)
            except _Deferred:
                deferred[si] = frames
                return
            except Exception as e:  # never abort the whole book for one bad page
                log.exception("segment %d failed", si)
                rs = [PageResult(0, si, seg.best, seg.t0, flags=seg.flags + ["processing_error"],
                                 info={"error": repr(e)})]
                if frames:
                    rs[0]._image = next(iter(frames.values()))
            if replaces is not None:
                for k in replaces[3]:
                    results[k] = None
                kept_refs[:] = [e for e in kept_refs if e is not replaces]
                log.info("segment %d replaces duplicate %d", si, replaces[0])
            idxs = []
            for r in rs:
                results.append(r); idxs.append(len(results) - 1)
            if thumb is not None:
                kept_refs.append((si, thumb, max(seg.scores), idxs))

        def enough():
            return cfg.max_pages > 0 and sum(r is not None for r in results) >= cfg.max_pages

        say("pages", 0.0)
        for idx, frame in src.iter_fetch(owner.keys()):
            frame = _resize_long(frame, cfg.max_frame_side)
            for si in owner[idx]:
                buf.setdefault(si, {})[idx] = frame
                if idx >= last_needed[si] and si not in done:
                    done.add(si)
                    finish(si)
            if deferred and prior.quads and prior.gutters:
                for si in sorted(deferred):
                    finish(si, allow_defer=False)
            if enough():
                log.info("Reached max_pages=%d; stopping early", cfg.max_pages)
                break
        else:
            for si in range(len(segs)):  # segments whose last frame failed to decode
                if si not in done and not enough():
                    done.add(si)
                    finish(si)
        for si in sorted(deferred):  # no page in the video was found by its text
            finish(si, allow_defer=False)

    pages = [r for r in results if r is not None]
    pages.sort(key=lambda r: (r.frame_index, r.info.get("half", "")))
    if cfg.max_pages > 0:
        pages = pages[:cfg.max_pages]  # a spread can overshoot by one
    for n, r in enumerate(pages, 1):
        r.page_no = n
        img = getattr(r, "_image", None)
        if img is not None:
            r.image_path = os.path.join(out_pages, f"page_{n:04d}.jpg")
            cv2.imwrite(r.image_path, img, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
            with open(os.path.join(out_pages, f"page_{n:04d}.txt"), "w", encoding="utf-8") as f:
                f.write(r.text)
    meta = {"video": os.path.abspath(video_path), "fps": fps, "segments": len(segs),
            "seconds": round(time.time() - t_start, 1), "config": cfg.to_dict()}
    write_outputs(out_dir, pages, meta, cfg.pdf)
    say("done", 1.0)
    return pages
