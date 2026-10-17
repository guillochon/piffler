"""Locate the page(s) in a frame, rectify perspective, split two-page spreads."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .hands import skin_mask

log = logging.getLogger(__name__)


def _order(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _text_density_map(img: np.ndarray, long_side: int = 1600) -> np.ndarray:
    """Boolean map of small, thin dark marks (letters) at `long_side` resolution."""
    s = min(1.0, long_side / max(img.shape[:2]))
    g = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    n, lbl, st, _ = cv2.connectedComponentsWithStats((bh > 30).astype(np.uint8), 8)
    small = np.zeros(n, bool)
    # words, not dots: a strainer's mesh or speckled stone gives round marks
    small[1:] = (st[1:, 2] <= 40) & (st[1:, 3] <= 20) & (st[1:, 4] >= 3) & (st[1:, 2] >= 1.6 * st[1:, 3])
    return small[lbl]


def _textiest_component(small: np.ndarray, L: np.ndarray, skin: Optional[np.ndarray],
                        txt: np.ndarray, prior: Optional[np.ndarray] = None):
    """Bright, solid region with the most letter-like marks inside.

    Pans, stovetops and white counters are as bright as paper but carry no text,
    and a speckled counter merges with the page at low thresholds, so several
    thresholds are tried and each region is judged by what's printed on it.
    A page with almost no print (title, blank, photo) is instead matched to
    `prior`, the mask of where the book was on neighbouring pages.
    Returns (mask or None, how it was chosen: "text" | "prior").
    """
    sh, sw = L.shape
    hi = float(np.percentile(L, 99))
    otsu, _ = cv2.threshold(cv2.GaussianBlur(L, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(5, int(0.02 * max(sh, sw)) | 1)
    best, best_score = None, 0.0
    near, near_score = None, 0.0
    for t in sorted({hi - d for d in (15, 25, 35, 50, 70)} | {float(otsu)}):
        b = (L > t).astype(np.uint8) * 255
        if skin is not None:
            b[skin > 0] = 0
        # letters sit inside bright paper; speckle on a dark counter doesn't
        surround = cv2.resize(cv2.blur(b, (5, 5)), (txt.shape[1], txt.shape[0]), interpolation=cv2.INTER_LINEAR)
        marks = txt & (surround > 160)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        n, lbl, st, _ = cv2.connectedComponentsWithStats(b, 8)
        for i in range(1, n):
            area = st[i, 4] / float(sh * sw)
            if area < 0.03:
                continue
            comp = (lbl == i).astype(np.uint8)
            cnt = max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
            hull = cv2.convexHull(cnt)
            solid = st[i, 4] / max(1.0, cv2.contourArea(hull))
            hm = np.zeros_like(comp)
            cv2.fillPoly(hm, [hull], 1)
            hm = cv2.resize(hm, (txt.shape[1], txt.shape[0]), interpolation=cv2.INTER_NEAREST)
            dens = float(marks[hm > 0].mean()) if hm.any() else 0.0
            log.debug("t=%.0f comp %s area=%.3f dens=%.4f solid=%.2f", t, st[i, :4].tolist(), area, dens, solid)
            if solid < 0.6:  # a page merged with its surroundings
                continue
            if prior is not None:
                hs = cv2.resize(hm, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
                iou = (hs & prior).sum() / max(1, (hs | prior).sum())
                if iou > 0.6 and iou * solid > near_score:
                    near, near_score = comp * 255, iou * solid
            if dens < 0.004:  # too little print to go on
                continue
            score = dens * np.sqrt(area) * solid ** 2
            if score > best_score:
                best, best_score = comp * 255, score
    if best is not None:
        return best, "text"
    return near, "prior"


def detect_page(img: np.ndarray, prior: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (quad[4x2] in img coords, page_mask uint8 at img size, flags).

    `prior` is an optional page mask at img size from neighbouring frames.
    Flags say how the page was found: by its text (no flag), near the prior
    ("page_near_neighbors"), or by brightness alone ("page_by_brightness").
    """
    h, w = img.shape[:2]
    flags: List[str] = []
    s = min(1.0, 800.0 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    full_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)

    L = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0]
    try:
        sk = skin_mask(small)
    except Exception:
        sk = None
    pr = None
    if prior is not None:
        pr = cv2.resize(prior, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    comp, how = _textiest_component(small, L, sk, _text_density_map(img), pr)
    if comp is not None and how == "prior":
        flags.append("page_near_neighbors")
    if comp is None:
        flags.append("page_by_brightness")  # weak evidence: callers may prefer a prior
        comp = _brightest_component(L, sk, flags)
    if comp is None:
        return full_quad, np.full((h, w), 255, np.uint8), flags
    return _fit_quad(comp, s, w, h, flags)


def _brightest_component(L: np.ndarray, sk: Optional[np.ndarray], flags: List[str]) -> Optional[np.ndarray]:
    """Fallback for pages without text: largest bright blob, preferring the centre."""
    sh, sw = L.shape
    k = max(5, int(0.03 * max(sh, sw)) | 1)
    # close removes dark text so the page becomes one bright blob
    Lc = cv2.morphologyEx(L, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    Lc = cv2.GaussianBlur(Lc, (5, 5), 0)
    t, bright = cv2.threshold(Lc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if Lc.std() < 8:  # uniform frame: page fills it (or nothing to find)
        bright[:] = 255
    if sk is not None:
        bright[sk > 0] = 0
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    if n <= 1:
        flags.append("page_not_found")
        return None
    # largest component, preferring ones near the centre
    cx, cy = sw / 2, sh / 2
    best, best_score = 1, -1
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        dist = np.hypot(x + bw / 2 - cx, y + bh / 2 - cy) / np.hypot(cx, cy)
        sc = a * (1.2 - dist)
        if sc > best_score:
            best, best_score = i, sc
    comp = (lbl == best).astype(np.uint8) * 255
    area_frac = comp.mean() / 255.0
    if area_frac < 0.12:
        flags.append("page_not_found")
        return None
    return comp


def _straightest_line(pts: np.ndarray, tol: float, rng: np.random.Generator):
    """RANSAC: the line through the most points (a page edge, not the hand on it)."""
    best, best_n = None, 0
    for _ in range(200):
        a, b = pts[rng.choice(len(pts), 2, replace=False)]
        d = b - a
        n = np.hypot(*d)
        if n < 1e-6:
            continue
        dist = np.abs(d[0] * (pts[:, 1] - a[1]) - d[1] * (pts[:, 0] - a[0])) / n
        inl = dist < tol
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best is None:
        return None, 0
    vx, vy, x0, y0 = cv2.fitLine(pts[best], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return (np.array([x0, y0]), np.array([vx, vy])), best_n


def _refine_sides(quad: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Move each side of `quad` onto the straight part of the outline near it.

    A hand gripping a page corner merges into the page blob and drags the
    hull's corner outward; the paper's own edges stay straight on either side.
    """
    if len(contour) < 20:
        return quad
    # resample the outline evenly so long straight runs aren't under-represented
    seg = np.diff(np.vstack([contour, contour[:1]]), axis=0)
    lens = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0], np.cumsum(lens)])
    ts = np.arange(0, cum[-1], 1.0)
    idx = np.clip(np.searchsorted(cum, ts, side="right") - 1, 0, len(contour) - 1)
    frac = ((ts - cum[idx]) / np.maximum(lens[idx], 1e-6))[:, None]
    pts = contour[idx] + seg[idx] * frac
    rng = np.random.default_rng(0)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        L = float(np.hypot(*d))
        if L < 10:
            return quad
        u = d / L
        rel = pts - a
        along = rel @ u
        off = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
        near = pts[(off < 0.2 * L) & (along > -0.1 * L) & (along < 1.1 * L)]
        line, n = _straightest_line(near, max(1.5, 0.008 * L), rng) if len(near) >= 10 else (None, 0)
        if line is None or n < 0.35 * L:
            lines.append((a, u))  # not enough straight edge seen: keep this side
        else:
            lines.append(line)
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return quad
        t = np.linalg.solve(A, p2 - p1)
        out.append(p1 + t[0] * d1)
    out = np.array(out, np.float32)
    area0, area1 = cv2.contourArea(quad), cv2.contourArea(out)
    if not cv2.isContourConvex(out.reshape(-1, 1, 2)) or not 0.6 * area0 < area1 < 1.15 * area0:
        return quad
    return out


def _fit_quad(comp: np.ndarray, s: float, w: int, h: int,
              flags: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    quad = None
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap) and cv2.contourArea(ap) > 0.8 * hull_area:
            quad = ap.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        # extreme points along the diagonals: robust for curved/irregular outlines
        p = hull.reshape(-1, 2).astype(np.float32)
        s_, d_ = p.sum(1), p[:, 0] - p[:, 1]
        quad = np.array([p[np.argmin(s_)], p[np.argmax(d_)], p[np.argmax(s_)], p[np.argmin(d_)]], np.float32)
        if cv2.contourArea(quad) < 0.7 * hull_area:
            x, y, bw, bh = cv2.boundingRect(hull)
            quad = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], np.float32)
            flags.append("page_quad_approx")

    # a page touching most of the frame border is cropped by the camera
    border = np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]])
    if (border > 0).mean() > 0.5:
        flags.append("page_touches_frame_edge")
    quad = _refine_sides(_order(quad), cnt.reshape(-1, 2).astype(np.float32))
    quad = _order(quad)
    # grow the quad ~1.5% outward so curled page edges don't clip text
    c = quad.mean(0)
    quad = c + (quad - c) * 1.03
    quad = quad / s
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    mask = cv2.resize(cv2.dilate(comp, np.ones((3, 3), np.uint8)), (w, h), interpolation=cv2.INTER_NEAREST)
    return quad, mask, flags


def rectify(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    W = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    H = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    W, H = max(W, 16), max(H, 16)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32),
                                    np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]))
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def find_gutter(img: np.ndarray, lo_frac: float = 0.33, hi_frac: float = 0.67) -> Tuple[Optional[int], float]:
    """x of the binding between two pages, and a confidence in [0, 1]."""
    h, w = img.shape[:2]
    s = min(1.0, 600.0 / w)
    g = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gh, gw = g.shape
    band = g[int(gh * 0.15): int(gh * 0.85)]
    k = max(3, int(gw * 0.02) | 1)
    shade = cv2.morphologyEx(band, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(np.float32)
    colshade = shade.mean(0)
    flat = band.astype(np.float32) / (shade + 1)
    ink = (flat < 0.75).mean(0)
    lo, hi = int(gw * lo_frac), int(gw * hi_frac)
    if hi - lo < 5:
        return None, 0.0
    darkness = (np.median(colshade) - colshade) / (colshade.std() + 1e-3)
    blank = 1.0 - ink / (ink.max() + 1e-6)
    score = cv2.GaussianBlur((0.6 * darkness + 0.8 * blank).reshape(1, -1).astype(np.float32), (0, 0), 2).ravel()
    x = lo + int(np.argmax(score[lo:hi]))
    # confidence: blank gutter between two inked halves, ideally shaded
    left_ink, right_ink = ink[:x].mean(), ink[x:].mean()
    c_blank = blank[max(0, x - 2): x + 3].mean()
    conf = float(np.clip(c_blank, 0, 1) * (1.0 if min(left_ink, right_ink) > 0.01 else 0.3)
                 * np.clip(0.5 + 0.25 * darkness[x], 0.3, 1.0))
    return int(round(x / s)), conf


def split_pages(img: np.ndarray, layout: str = "auto",
                hint: Optional[float] = None) -> Tuple[List[np.ndarray], List[str], Optional[float]]:
    """Split a spread at its gutter. Returns (pages, flags, gutter x / width
    when it was found confidently, else None).

    `hint` is where the gutter sat (as a fraction of width) on confidently
    split spreads earlier in the same book; it settles title pages, blank
    pages and photos, whose gutter is hard to see on its own.
    """
    h, w = img.shape[:2]
    aspect = w / float(h)
    flags: List[str] = []
    if layout == "single":
        return [img], flags, None
    x, conf = find_gutter(img)
    log.debug("gutter own x=%s (%.3f) conf=%.2f hint=%s", x, (x or 0) / float(w), conf, hint)
    # a photo's edge or a blank column can pose as the binding, so a gutter far
    # from where this book's gutter usually sits is looked for again near it
    if hint is not None and (conf < 0.5 or x is None or abs(x / float(w) - hint) > 0.08):
        x2, conf2 = find_gutter(img, max(0.2, hint - 0.06), min(0.8, hint + 0.06))
        log.debug("gutter near hint: x=%s (%.3f) conf=%.2f", x2, (x2 or 0) / float(w), conf2)
        if x2 is None:
            x, conf = int(round(hint * w)), 0.0
        else:  # best spot near the usual gutter, trusted only as far as it's visible
            wgt = min(1.0, conf2 / 0.35)
            x, conf = int(round(hint * w + wgt * (x2 - hint * w))), conf2
        if conf < 0.35:
            flags.append("gutter_from_neighbors")
    # pocket books have narrow pages, so a spread can be nearly square; a
    # single page is rarely wider than ~0.8, so lean on the gutter above that
    is_spread = (layout == "spread" or hint is not None or aspect > 1.3
                 or (aspect > 1.15 and conf > 0.35) or (aspect > 0.9 and conf > 0.5))
    if not is_spread:
        return [img], flags, None
    if x is None:
        x = w // 2
        flags.append("gutter_guessed")
    elif conf < 0.35 and "gutter_from_neighbors" not in flags:
        flags.append("gutter_uncertain")
    overlap = int(0.015 * w)  # pocket books print close to the binding
    sure = x / float(w) if conf >= 0.5 else None
    return [img[:, : x + overlap], img[:, max(0, x - overlap):]], flags, sure
