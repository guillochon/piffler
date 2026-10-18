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
from .dewarp import deshear, dewarp, find_lines, flatten_illumination, text_mask
from .export import write_outputs
from .hands import align_to, occluder_mask, remove_fingers, skin_mask
from .ocr import OCREngine, Word, best_rotation, rotate, words_to_text
from .page import detect_page, rectify, split_pages
from .repeats import drop_repeats
from .scan import clean_scan
from .segment import FrameStats, Segment, analyze, find_segments
from .video import VideoSource

log = logging.getLogger(__name__)

# flags that mean "look at this page": something may be missing or wrong on it.
# Informational flags (no_text_found on a blank page, short_hold, gutter_from_neighbors)
# describe how a page was made and are not reasons to re-shoot it.
REVIEW_FLAGS = {"finger_may_cover_text", "low_quality_frame", "page_not_found", "page_from_neighbors",
                "processing_error", "gutter_guessed", "no_still_segments", "frame_decode_failed",
                "possible_duplicate", "low_ocr_confidence", "dewarp_failed"}


@dataclass
class PageResult:
    page_no: int
    segment: int
    frame_index: int
    time_s: float
    image_path: str = ""   # the clean scan (or the page photo with --no-scan)
    photo_path: str = ""   # the flattened page photo
    image_scale: float = 1.0  # image pixels per OCR-box pixel
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
            "photo": os.path.relpath(self.photo_path, root) if root and self.photo_path else self.photo_path,
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

    def steady(self) -> bool:
        """False when recent confident pages put the book in different places:
        a hand-held phone, where a borrowed location would be wrong."""
        if len(self.quads) < 3:
            return True
        Q = np.stack(self.quads)
        diag = float(np.hypot(*(Q[-1].max(0) - Q[-1].min(0))))
        return float(np.median(np.hypot(*(Q - np.median(Q, axis=0)).transpose(2, 0, 1)))) < 0.06 * diag

    def get(self, shape):
        if not self.quads or not self.steady():
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


def _text_zone(crop: np.ndarray, region: np.ndarray) -> np.ndarray:
    """The page's text block: the box around its print, so a finger hiding
    words still counts as covering text."""
    tm = (text_mask(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) > 0) & (region > 0)
    zone = np.zeros(crop.shape[:2], np.uint8)
    ys, xs = np.nonzero(tm)
    if len(xs) < 50:
        zone[region > 0] = 255
        return zone
    x0, x1 = np.percentile(xs, [1, 99]).astype(int)
    y0, y1 = np.percentile(ys, [1, 99]).astype(int)
    zone[y0:y1 + 1, x0:x1 + 1] = 255
    return zone & region


def _column_zone(img: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Where text runs: each text column's box, from its detected lines (left
    and right edges = typical line start and end; top and bottom = first and
    last line, half a line pitch out). Margins, page edges and blank pages are
    outside it. A spread gives two columns."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = min(1.0, 1400.0 / g.shape[0])
    gs = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    zone = np.zeros(gs.shape, np.uint8)
    lines = find_lines(text_mask(gs))
    if len(lines) >= 3:
        spans = np.array([(l[:, 0].min(), l[:, 0].max(), float(np.median(l[:, 1]))) for l in lines], np.float32)
        cx = (spans[:, 0] + spans[:, 1]) / 2
        cols = [spans]
        order = np.sort(cx)
        if len(order) >= 6:  # two columns if line centres split around a wide gap
            gaps = np.diff(order)
            k = int(np.argmax(gaps))
            if gaps[k] > 0.15 * gs.shape[1]:
                cut = (order[k] + order[k + 1]) / 2
                cols = [spans[cx < cut], spans[cx >= cut]]
        for c in cols:
            if len(c) < 3:
                continue
            ys = np.sort(c[:, 2])
            pitch = float(np.median(np.diff(ys))) if len(ys) > 1 else 20.0
            x0, x1 = np.median(c[:, 0]), np.median(c[:, 1])
            y0, y1 = ys[0] - 0.5 * pitch, ys[-1] + 0.5 * pitch
            zone[int(max(0, y0)):int(y1) + 1, int(max(0, x0)):int(x1) + 1] = 255
    zone = cv2.resize(zone, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(zone, region)


def _finger_cover(crop: np.ndarray, region: np.ndarray, zone: np.ndarray) -> float:
    """Share of the text block hidden by hands in this frame (judged small)."""
    s = min(1.0, 800.0 / max(crop.shape[:2]))
    c = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    r = cv2.resize(region, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST)
    z = cv2.resize(zone, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    sk = skin_mask(c, r, pale=True) > 0
    return float((sk & z).sum()) / max(1, int(z.sum()))


def _consensus_cover(pool: Dict[int, np.ndarray], order: List[int], zone: np.ndarray) -> Dict[int, float]:
    """Share of the text block where each frame disagrees with the others.

    Skin colour alone misses a hand now and then (blur, glare, pale skin under
    cool light), and that frame then looks finger-free. Aligned to each other,
    the frames of one page agree except where something covers the page in only
    some of them, so a per-pixel median shows what the page looks like and big
    blobs that differ from it are hands, whatever their colour.
    """
    anchor = pool[order[0]]
    s = min(1.0, 600.0 / max(anchor.shape[:2]))
    small = {i: cv2.resize(pool[i], None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for i in order}
    a = small[order[0]]
    h, w = a.shape[:2]
    grays = {order[0]: cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)}
    for i in order[1:]:
        H = align_to(a, small[i])
        if H is not None:
            grays[i] = cv2.warpPerspective(cv2.cvtColor(small[i], cv2.COLOR_BGR2GRAY), H, (w, h),
                                           borderMode=cv2.BORDER_REPLICATE)
    if len(grays) < 3:
        return {}
    z = cv2.resize(zone, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    if not z.any():
        return {}
    ids = list(grays)
    stack = np.stack([cv2.GaussianBlur(grays[i], (5, 5), 0) for i in ids]).astype(np.float32)
    stack *= (np.median(stack[:, z]) + 1) / (np.median(stack[:, z], axis=1) + 1)[:, None, None]  # exposure
    med = np.median(stack, axis=0)
    paper_lvl = float(np.percentile(med[z], 90))  # differences count relative to the paper's brightness
    k = np.ones((5, 5), np.uint8)  # wider than a misaligned stroke edge, narrower than a finger
    out = {}
    for i, g in zip(ids, stack):
        dev = cv2.morphologyEx(((np.abs(g - med) > 0.12 * paper_lvl) & z).astype(np.uint8), cv2.MORPH_OPEN, k)
        out[i] = float(dev[z].mean())
    return out


def _process_segment(si: int, seg: Segment, frames: Dict[int, np.ndarray], cfg: Config,
                     engine: Optional[OCREngine], orient: _Orientation, fps: float,
                     out_pages: str, debug_dir: Optional[str], prior: _PagePrior,
                     allow_defer: bool, extra=None) -> List[PageResult]:
    """`extra` is an earlier hold of the same page, (first frame index, frame ids
    best-first, frames): its frames
    join the pool, so the frame least covered by fingers can come from either
    hold and the other hold's frames can fill in where fingers were."""
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
    holds = [(quad * ds, pmask, cands, frames)]  # (quad, page mask at det scale, frame ids, frames)
    if extra is not None:
        estart, eids, eframes = extra
        ecands = [i for i in eids if i in eframes and i not in frames]
        if ecands:
            edet = _resize_long(eframes[ecands[0]], cfg.detect_long_side)
            equad, emask, eflags = detect_page(edet, pmask)
            if "page_by_brightness" in eflags:  # the book barely moved between holds
                equad, emask = quad, pmask
            holds.append((equad * ds, emask, ecands, eframes))
            info["merged_hold"] = int(estart)
            flags = [f for f in flags if f != "short_hold"]
    H, W = ref.shape[:2]
    allq = np.concatenate([h[0] for h in holds])
    x0, y0 = allq.min(0)
    x1, y1 = allq.max(0)
    pad = 0.04 * max(x1 - x0, y1 - y0)
    x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    x1, y1 = int(min(W, x1 + pad + 1)), int(min(H, y1 + pad + 1))
    cs = min(1.0, cfg.work_long_side / float(max(x1 - x0, y1 - y0)))

    def crop(f):
        f = f[y0:y1, x0:x1]
        return cv2.resize(f, None, fx=cs, fy=cs, interpolation=cv2.INTER_AREA) if cs < 1 else f

    pool, where = {}, {}  # cropped frames, and which hold each came from
    for hi_, (_, _, ids, src_frames) in enumerate(holds):
        for i in ids:
            pool[i], where[i] = crop(src_frames[i]), hi_
    crops = []
    for q, m, _, _ in holds:
        cq = ((q - [x0, y0]) * cs).astype(np.float32)
        cm = cv2.resize(cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)[y0:y1, x0:x1],
                        (pool[ref_idx].shape[1], pool[ref_idx].shape[0]), interpolation=cv2.INTER_NEAREST)
        crops.append((cq, cm))
    # reference = the frame whose text is least hidden by fingers; ties go to
    # the sharpest (candidates come sharpest-first)
    zones = [_text_zone(pool[h[2][0]], crops[k][1]) for k, h in enumerate(holds)]
    order = [i for h in holds for i in h[2]]
    cover = {i: _finger_cover(pool[i], crops[where[i]][1], zones[where[i]]) for i in order}
    if len(order) >= 3:
        agree = _consensus_cover(pool, order, zones[0])
        for i in order:  # a frame that won't align with the rest can't be vouched for
            cover[i] = max(cover[i], agree.get(i, cover[i] + 0.01)) if agree else cover[i]
    ref_idx = min(order, key=lambda i: (round(cover[i], 3), order.index(i)))
    info["finger_cover"] = round(cover[ref_idx], 4)
    if debug_dir:  # every pooled frame with its hands (red), text block (green) and score
        tiles = []
        for i in order:
            c, (_, m) = pool[i], crops[where[i]]
            s_ = 400.0 / max(c.shape[:2])
            cs_ = cv2.resize(c, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA)
            ms_ = cv2.resize(m, (cs_.shape[1], cs_.shape[0]), interpolation=cv2.INTER_NEAREST)
            sk_ = skin_mask(cs_, ms_, pale=True) > 0
            cs_[sk_] = (0.4 * cs_[sk_] + [0, 0, 153]).astype(np.uint8)
            z_ = cv2.resize(zones[where[i]], (cs_.shape[1], cs_.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.drawContours(cs_, cv2.findContours(z_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 180, 0), 1)
            cv2.putText(cs_, f"{i}{'*' if i == ref_idx else ''} {cover[i]:.3f}", (4, 16), 0, 0.5, (255, 0, 0), 1)
            tiles.append(cs_)
        hh = min(t.shape[0] for t in tiles)
        cv2.imwrite(os.path.join(debug_dir, f"seg{si:04d}_pool.jpg"), np.hstack([t[:hh] for t in tiles]))
    ref = pool[ref_idx]
    quad, pmask = crops[where[ref_idx]]
    frames, cands = pool, [ref_idx] + [i for i in order if i != ref_idx]

    clean = ref
    covered_pts = None
    if cfg.remove_fingers:
        others = [frames[i] for i in cands[1:]]
        occ = occluder_mask(ref, others, pmask)  # colour-free: catches what skin colour misses
        clean, residual, fstats = remove_fingers(ref, others, pmask, cfg.skin_paper_delta, extra=occ)
        fstats["occluder_frac"] = float((occ > 0).sum()) / max(1, int((pmask > 0).sum()))
        info["fingers"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in fstats.items()}
        if residual.any():
            # only skin no other frame could fill gets painted with paper; it hides
            # text only where it lies inside the text block (margins don't matter)
            zone = _column_zone(clean, pmask) > 0
            cov = (residual > 0) & zone
            zone_px = max(1, int(zone.sum()))
            info["hidden_text_frac"] = round(float(cov.sum()) / zone_px, 4) if zone.any() else 0.0
            if cov.sum() > 0.002 * zone_px:
                ys, xs = np.nonzero(cov)
                covered_pts = np.float32(np.stack([xs, ys], 1))  # assigned to a page after the split
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"seg{si:04d}_ref.jpg"), _resize_long(ref, 1200))
            cv2.imwrite(os.path.join(debug_dir, f"seg{si:04d}_clean.jpg"), _resize_long(clean, 1200))

    region = rectify(clean, quad)
    covered_x = None
    if covered_pts is not None:  # where, across the rectified spread, fingers may hide text
        rh, rw = region.shape[:2]
        M = cv2.getPerspectiveTransform(quad.astype(np.float32),
                                        np.float32([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]]))
        covered_x = cv2.perspectiveTransform(covered_pts.reshape(-1, 1, 2), M).reshape(-1, 2)[:, 0]
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
        if covered_x is not None:
            if len(halves) == 1 or rot:
                pf.append("finger_may_cover_text")
            else:  # only the page the hidden text is on
                gx = halves[0].shape[1] - int(0.015 * region.shape[1])  # the split line
                n_here = int((covered_x < gx).sum() if hi == 0 else (covered_x >= gx).sum())
                if n_here > 0.001 * zone_px:
                    pf.append("finger_may_cover_text")
        pinfo = dict(info)
        if cfg.dewarp:
            try:
                page, dinfo = dewarp(page)
                pinfo["dewarp"] = dinfo
            except Exception as e:
                log.warning("dewarp failed on segment %d: %s", si, e)
                pf.append("dewarp_failed")
        if debug_dir:  # before lighting correction, to tune what comes after
            cv2.imwrite(os.path.join(debug_dir, f"seg{si:04d}_{hi}_raw.png"), page)
        page = flatten_illumination(page)
        if cfg.dewarp:  # level lines can still sit in a leaning column; judged on even light
            try:
                page, sinfo = deshear(page)
                pinfo.setdefault("dewarp", {}).update(sinfo)
            except Exception as e:
                log.warning("deshear failed on segment %d: %s", si, e)
        words: List[Word] = []
        if engine is not None:
            words = engine(page)
            if len(halves) > 1:
                # each half overlaps the facing page a little; words lying wholly in
                # that strip are the facing page's line starts or ends
                strip, pw = 2 * int(0.015 * region.shape[1]), page.shape[1]
                words = [w for w in words if (w.x0 <= pw - strip if hi == 0 else w.box[:, 0].max() >= strip)]
                words, cut = _drop_facing_column(words, pw, gutter_right=(hi == 0))
                if cut is not None:
                    pinfo["facing_cut_px"] = cut
            if not words:
                pf.append("no_text_found")
            elif np.mean([w.score for w in words]) < 0.75:
                pf.append("low_ocr_confidence")
        r = PageResult(0, si, ref_idx, ref_idx / fps, words=words, flags=pf, info=pinfo)
        r.text = words_to_text(words)
        r._image = page
        if len(halves) > 1:
            r.info["half"] = "left" if hi == 0 else "right"
            r.info["overlap_px"] = int(0.015 * region.shape[1])
        results.append(r)
    return results


def _paper_tone(img: np.ndarray) -> np.ndarray:
    """The page's typical paper colour (BGR)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.median(img[g >= np.percentile(g, 70)], axis=0).astype(np.uint8)


def _drop_facing_column(words: List[Word], pw: int, gutter_right: bool):
    """Words in a narrow strip at the gutter, cut off from the page's own text
    by an empty margin, are the facing page's line ends or starts (the split
    landed a little off). Returns (kept words, x of that margin) - x is None
    when there is no such strip."""
    if not words:
        return words, None
    lh = float(np.median([w.height for w in words]))
    band = int(0.15 * pw)  # facing-page leftovers sit right at the gutter
    if len(words) < 12:
        # an otherwise blank page whose only "words" are a stack hard against the
        # gutter: the facing page's line starts or ends
        near = [w for w in words if (w.x0 > pw - band if gutter_right else w.box[:, 0].max() < band)]
        if len(near) == len(words) and len(near) >= 3:
            if gutter_right:
                return [], int(min(w.x0 for w in near) - 0.3 * lh)
            return [], int(max(w.box[:, 0].max() for w in near) + 0.3 * lh)
        return words, None
    occ = np.zeros(pw + 1, np.int32)  # how many words cover each x
    for w in words:
        a, b = int(max(0, w.x0)), int(min(pw, w.box[:, 0].max()))
        occ[a:b + 1] += 1
    prof = occ[pw - band:][::-1] if gutter_right else occ[:band]  # walking inward from the gutter
    # the margin: the first long empty run after some ink, walking from the gutter
    inked, run, start, found = False, 0, None, False
    for k, v in enumerate(prof):
        if v > 0:
            if inked and run >= 0.8 * lh:
                found = True
                break
            inked, run, start = True, 0, None
        elif inked:
            start = k if start is None else start
            run += 1
    found = found or (inked and run >= 0.8 * lh)  # the page's own text starts past the band
    if not found:
        return words, None
    gap_mid = start + run // 2
    cut = (pw - gap_mid) if gutter_right else gap_mid
    stray = [w for w in words if (w.x0 > cut if gutter_right else w.box[:, 0].max() < cut)]
    if not stray or len(stray) > 0.3 * len(words):
        return words, None
    keep = [w for w in words if all(w is not x for x in stray)]
    return keep, int(cut)


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
                      cfg.analysis_fps, cfg.analysis_width, cfg.rotate, "analysis-v2"))

    with VideoSource(video_path, rotate=cfg.rotate, max_bad_reads=cfg.max_bad_reads, use_gpu=cfg.gpu) as src:
        fps = src.info.fps
        log.info("Video: %dx%d @ %.2f fps, decoded with %s", src.info.width, src.info.height, fps, src.decoder)
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

        engine = OCREngine(cfg.lang_det_long_side, cfg.ocr_min_score, cfg.gpu) if cfg.ocr else None
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

        def finish(si, allow_defer=True, repeats_only=False):
            """Process segment `si`; returns False when `repeats_only` and it isn't
            a repeat hold of the latest page (a new page past --max-pages)."""
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
                    psi, pthumb, pscore, pidx = entry[:4]
                    # like every hold already pooled for that page, not just the latest:
                    # a hold caught mid-turn resembles both pages and would chain them
                    pool_thumbs = entry[6] if len(entry) > 6 else [pthumb]
                    sim = min(page_similarity(thumb, t) for t in pool_thumbs)
                    log.debug("segment %d vs %d: sim %.3f (over %d holds)", si, psi, sim, len(pool_thumbs))
                    if sim < cfg.dedupe_similarity:
                        continue
                    adjacent = any(entry is e for e in after) or (before and entry is before[0])
                    if adjacent and entry[5] is not None:
                        # same page held twice in a row: pool both holds' frames, so the
                        # frame least covered by fingers can come from either one
                        replaces = entry
                        log.info("segment %d repeats %d (sim %.2f); pooling their frames", si, psi, sim)
                    elif adjacent:  # no frames kept for the other hold: keep the better capture
                        if max(seg.scores) <= pscore:
                            log.info("segment %d duplicates %d (sim %.2f); skipped", si, psi, sim)
                            return True
                        replaces = entry
                    else:
                        seg.flags.append("possible_duplicate")
                    break
            if repeats_only and (replaces is None or replaces[5] is None):
                return False
            try:
                extra = (replaces[0], replaces[4], replaces[5]) if replaces is not None and replaces[5] else None
                rs = _process_segment(si, seg, frames, cfg, engine, orient, fps, out_pages, debug_dir,
                                      prior, allow_defer, extra)
            except _Deferred:
                deferred[si] = frames
                return True
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
                # keep this hold's frames (with any it pooled) for a repeat hold to
                # join; only for the latest pages, to bound memory
                ids = [i for i in seg.candidates if i in frames]
                pooled = dict(frames)
                if replaces is not None and replaces[5]:
                    ids += [i for i in replaces[4] if i not in pooled]
                    pooled.update(replaces[5])
                ids = ids[:12]
                pooled = {i: pooled[i] for i in ids}
                thumbs = ((list(replaces[6]) if replaces is not None and len(replaces) > 6 else
                           [replaces[1]] if replaces is not None else []) + [thumb])[-4:]
                kept_refs.append((si, thumb, max(seg.scores), idxs, ids, pooled, thumbs))
                latest = sorted(kept_refs, key=lambda e: -e[0])[:2]
                kept_refs[:] = [e if e[5] is None or any(e is k for k in latest) else e[:5] + (None,) + e[6:]
                                for e in kept_refs]
            return True

        def enough():
            return cfg.max_pages > 0 and sum(r is not None for r in results) >= cfg.max_pages

        say("pages", 0.0)
        full, stop = False, False
        for idx, frame in src.iter_fetch(owner.keys()):
            frame = _resize_long(frame, cfg.max_frame_side)
            for si in owner[idx]:
                buf.setdefault(si, {})[idx] = frame
                if idx >= last_needed[si] and si not in done:
                    done.add(si)
                    # past --max-pages, only repeat holds of the last page still help it
                    stop = not finish(si, repeats_only=full) or stop
            if deferred and prior.quads and (prior.gutters or not prior.steady()):
                for si in sorted(deferred):
                    finish(si, allow_defer=False)
            if enough() and not full:
                full = True
                log.info("Reached max_pages=%d; finishing the last page's repeat holds", cfg.max_pages)
            if stop:
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
    dropped = []
    if cfg.dedupe:
        pages, dropped = drop_repeats(pages)
        for d in dropped:
            log.info("dropped the %s page from frame %d: %s", d["half"] or "whole", d["frame_index"], d["reason"])
    if cfg.max_pages > 0:
        pages = pages[:cfg.max_pages]  # a spread can overshoot by one
    for n, r in enumerate(pages, 1):
        r.page_no = n
        img = getattr(r, "_image", None)
        if img is not None:
            cut = r.info.get("facing_cut_px")
            if cut is not None:  # the facing page's strip at the gutter becomes paper
                img = img.copy()
                if r.info.get("half") == "left":
                    img[:, cut:] = _paper_tone(img)
                else:
                    img[:, :cut] = _paper_tone(img)
            r.photo_path = os.path.join(out_pages, f"page_{n:04d}_photo.jpg")
            cv2.imwrite(r.photo_path, img, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
            r.image_path = r.photo_path
            if cfg.scan:
                try:
                    side = {"left": "right", "right": "left"}.get(r.info.get("half"))  # binding side
                    sc = clean_scan(img, upscale=cfg.scan_upscale, gutter=side,
                                    overlap=r.info.get("overlap_px", 0), style=cfg.scan_style,
                                    word_boxes=[w.box for w in r.words] or None)
                    if cfg.scan_style == "bw":  # flat tones: PNG is small and exact
                        r.image_path = os.path.join(out_pages, f"page_{n:04d}.png")
                        cv2.imwrite(r.image_path, sc)
                    else:  # natural tones and photos: JPEG keeps the PDF a sensible size
                        r.image_path = os.path.join(out_pages, f"page_{n:04d}.jpg")
                        cv2.imwrite(r.image_path, sc, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    r.image_scale = cfg.scan_upscale
                except Exception as e:  # the photo is still there
                    log.warning("scan cleanup failed on page %d: %s", n, e)
                    r.image_path = r.photo_path
            with open(os.path.join(out_pages, f"page_{n:04d}.txt"), "w", encoding="utf-8") as f:
                f.write(r.text)
    meta = {"video": os.path.abspath(video_path), "fps": fps, "segments": len(segs), "dropped": dropped,
            "seconds": round(time.time() - t_start, 1), "config": cfg.to_dict()}
    write_outputs(out_dir, pages, meta, cfg.pdf)
    say("done", 1.0)
    return pages
