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


def detect_page(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (quad[4x2] in img coords, page_mask uint8 at img size, flags)."""
    h, w = img.shape[:2]
    flags: List[str] = []
    s = min(1.0, 800.0 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    full_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)

    L = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0]
    k = max(5, int(0.03 * max(sh, sw)) | 1)
    # close removes dark text so the page becomes one bright blob
    Lc = cv2.morphologyEx(L, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    Lc = cv2.GaussianBlur(Lc, (5, 5), 0)
    t, bright = cv2.threshold(Lc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if Lc.std() < 8:  # uniform frame: page fills it (or nothing to find)
        bright[:] = 255
    try:
        sk = skin_mask(small)
        bright[sk > 0] = 0
    except Exception:
        pass
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    if n <= 1:
        flags.append("page_not_found")
        return full_quad, np.full((h, w), 255, np.uint8), flags
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
        return full_quad, np.full((h, w), 255, np.uint8), flags

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


def find_gutter(img: np.ndarray) -> Tuple[Optional[int], float]:
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
    lo, hi = int(gw * 0.33), int(gw * 0.67)
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


def split_pages(img: np.ndarray, layout: str = "auto") -> Tuple[List[np.ndarray], List[str]]:
    h, w = img.shape[:2]
    aspect = w / float(h)
    flags: List[str] = []
    if layout == "single":
        return [img], flags
    x, conf = find_gutter(img)
    is_spread = layout == "spread" or (aspect > 1.15 and (conf > 0.35 or aspect > 1.3))
    if not is_spread:
        return [img], flags
    if x is None:
        x = w // 2
        flags.append("gutter_guessed")
    elif conf < 0.35:
        flags.append("gutter_uncertain")
    overlap = int(0.005 * w)
    return [img[:, : x + overlap], img[:, max(0, x - overlap):]], flags
