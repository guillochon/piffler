"""Flatten curved book pages using the shape of the text lines.

Each text line is detected, its centre-line fitted with a low-order
polynomial, and a smooth 2-D vertical displacement field d(x, y) is fitted
to all lines by robust least squares. The page is then remapped so every
line becomes straight. The fit is clamped to where text was observed and the
result is rejected if it doesn't measurably straighten the lines, so bad
input (pictures, blank pages, blur) falls back to the undistorted crop.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def text_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    k = max(15, int(min(h, w) / 25) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    norm = cv2.divide(gray, bg, scale=255)
    blk = max(15, int(min(h, w) / 30) | 1)
    m = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blk, 12)
    m &= ((norm < 215).astype(np.uint8) * 255)
    # drop blobs touching the border (page edges, gutter shadow, fingers)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    bad = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h or bh > 0.08 * h or a < 4:
            bad[i] = True
    m[bad[lbl]] = 0
    return m


def find_lines(mask: np.ndarray) -> List[np.ndarray]:
    """Return list of (N,2) arrays of centre-line samples (x, y)."""
    h, w = mask.shape
    kx = max(9, int(w * 0.025))
    sm = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    sm = cv2.erode(sm, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, kx // 2), 1)))
    sm = cv2.morphologyEx(sm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(sm, 8)
    lines = []
    step = max(2, w // 300)
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        thick = a / float(bw)  # mean band thickness; robust to curved/tilted lines
        if bw < 0.12 * w or bh > 0.15 * h or thick < 2 or thick > 0.035 * h or bw / thick < 8:
            continue
        sub_lbl = lbl[y: y + bh, x: x + bw] == i
        pts = []
        for cx in range(0, bw, step):
            col = sub_lbl[:, cx]
            if col.sum() < 2:
                continue
            # centre of the line band (more stable than ink centroid of ascenders)
            yy = np.where(col)[0]
            pts.append((x + cx, y + 0.5 * (yy.min() + yy.max())))
        if len(pts) < 8:
            continue
        p = np.array(pts, np.float32)
        deg = 3 if (np.ptp(p[:, 0]) > 0.4 * w) else 2
        c = np.polyfit(p[:, 0], p[:, 1], deg)
        r = p[:, 1] - np.polyval(c, p[:, 0])
        if r.std() > 0.35 * bh + 1:
            continue
        xs = np.linspace(p[:, 0].min(), p[:, 0].max(), 40)
        lines.append(np.stack([xs, np.polyval(c, xs)], 1))
    return lines


def _design(xn, yn, dx=4, dy=3):
    cols = [xn ** a * yn ** b for a in range(dx + 1) for b in range(dy + 1)]
    return np.stack(cols, -1)


def _fit_field(lines, w, h):
    X, Y, D = [], [], []
    for ln in lines:
        ref = np.median(ln[:, 1])
        X.append(ln[:, 0]); Y.append(ln[:, 1]); D.append(ln[:, 1] - ref)
    X, Y, D = map(np.concatenate, (X, Y, D))
    xn, yn = X / w * 2 - 1, Y / h * 2 - 1
    A = _design(xn, yn)
    wts = np.ones_like(D)
    coef = None
    for _ in range(4):
        Aw = A * wts[:, None]
        lam = 1e-3 * len(D)
        coef = np.linalg.solve(Aw.T @ Aw + lam * np.eye(A.shape[1]), Aw.T @ (D * wts))
        r = D - A @ coef
        mad = np.median(np.abs(r)) * 1.4826 + 0.5
        wts = (np.abs(r) < 3 * mad).astype(np.float64)
    before = float(np.sqrt(np.mean(D ** 2)))
    after = float(np.sqrt(np.mean((D - A @ coef)[wts > 0] ** 2)))
    bounds = (X.min(), X.max(), Y.min(), Y.max(), float(np.abs(D).max()))
    return coef, before, after, bounds


def _deskew(img, lines):
    angs = []
    for ln in lines:
        c = np.polyfit(ln[:, 0], ln[:, 1], 1)
        angs.append(np.degrees(np.arctan(c[0])))
    a = float(np.median(angs)) if angs else 0.0
    return a


def dewarp(img: np.ndarray, work_height: int = 1400) -> Tuple[np.ndarray, dict]:
    info = {"dewarped": False, "lines": 0, "deskew_deg": 0.0, "curl_px": 0.0}
    H0, W0 = img.shape[:2]
    s = min(1.0, work_height / float(H0))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lines = find_lines(text_mask(g))
    info["lines"] = len(lines)
    if len(lines) < 3:
        info["reason"] = "too_few_text_lines"
        return img, info

    # 1) deskew if needed
    ang = _deskew(small, lines)
    if 0.4 < abs(ang) < 15:
        M = cv2.getRotationMatrix2D((W0 / 2, H0 / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (W0, H0), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        lines = find_lines(text_mask(g)) or lines
        info["deskew_deg"] = round(ang, 2)

    h, w = g.shape
    span = (max(l[:, 1].max() for l in lines) - min(l[:, 1].min() for l in lines)) / h
    if len(lines) < 3 or span < 0.15:
        info["reason"] = "text_lines_cover_too_little_page"
        return img, info
    coef, before, after, (x0, x1, y0, y1, dmax) = _fit_field(lines, w, h)
    info.update(lines=len(lines), curl_px=round(before / s, 1))
    if before < 1.0:
        info["reason"] = "already_flat"
        return img, info
    if after > 0.6 * before:
        info["reason"] = "fit_did_not_straighten"
        return img, info

    # 2) evaluate the field on a coarse grid (clamped to where text exists), upsample.
    #    Grid points sit where cv2.resize samples them, so upsampling is exact-aligned.
    Hc, Wc = max(2, H0 // 8), max(2, W0 // 8)
    gy = ((np.arange(Hc) + 0.5) * H0 / Hc - 0.5) * s   # in working (small) coordinates
    gx = ((np.arange(Wc) + 0.5) * W0 / Wc - 0.5) * s
    XX, YY = np.meshgrid(gx, gy)

    def field(x, y):
        xc = np.clip(x, x0, x1); yc = np.clip(y, y0, y1)
        d = _design(xc / w * 2 - 1, yc / h * 2 - 1) @ coef
        return np.clip(d, -1.5 * dmax, 1.5 * dmax)

    d1 = field(XX, YY)
    d2 = field(XX, YY + d1)  # one fixed-point iteration: d is defined at source y
    dfull = cv2.resize((d2 / s).astype(np.float32), (W0, H0), interpolation=cv2.INTER_LINEAR)
    mx, my = np.meshgrid(np.arange(W0, dtype=np.float32), np.arange(H0, dtype=np.float32))
    my = my + dfull
    out = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    info.update(dewarped=True, residual_px=round(after / s, 2))
    return out, info


def flatten_illumination(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Even out shadows (gutter shading, uneven lamp light) while keeping colour.
    Pictures are left as they are: the paper's light is carried across them."""
    from .scan import paper_background, picture_mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pics = picture_mask(gray)
    bg = paper_background(img, pics, long_side=600)
    paper_px = bg[~pics] if (~pics).any() else bg.reshape(-1, 3)
    target = np.percentile(paper_px.reshape(-1, 3), 90, axis=0)
    # lift underexposed pages so paper ends up near white, keeping its tint
    lum = float(target.mean())
    if lum < 225:
        target = target * min(2.5, 235.0 / max(lum, 1.0))
    target = np.minimum(target, 250)
    gain = target / np.maximum(bg, 12)
    gain = 1 + strength * (gain - 1)
    out = img.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)




def _edge_curve(x: np.ndarray, y: np.ndarray, deg: int = 2):
    """Robust smooth curve x = f(y) through a column edge, ignoring indents and
    short lines (outliers). Returns (poly coeffs, median residual, inliers) or None."""
    if len(x) < 8:
        return None
    deg = deg if len(x) >= 12 else 1
    c = np.polyfit(y, x, deg)
    keep = np.ones(len(x), bool)
    for _ in range(3):
        r = np.abs(x - np.polyval(c, y))
        keep = r < max(2.0, 2.0 * np.median(r))
        if keep.sum() < 6:
            return None
        c = np.polyfit(y[keep], x[keep], deg)
    r = np.abs(x[keep] - np.polyval(c, y[keep]))
    return c, float(np.median(r)), int(keep.sum())


def deshear(img: np.ndarray, work_height: int = 1400) -> Tuple[np.ndarray, dict]:
    """Make the text column upright and evenly wide (horizontal flattening).

    Straightening lines moves pixels up and down only, so a page that is curved
    or rectified slightly askew keeps level lines in a leaning, bowed column.
    In justified text every full line starts and ends on the column edges, so
    the left and right edges are fitted as smooth curves of height, and each
    row is stretched horizontally so both edges stand straight and vertical."""
    info = {"deshear_px": 0.0}
    H0, W0 = img.shape[:2]
    s = min(1.0, work_height / float(H0))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    tm = text_mask(g)
    lines = find_lines(tm)
    if len(lines) < 8:
        info["reason"] = "too_few_text_lines"
        return img, info
    hh = max(2, int(np.median([np.ptp(l[:, 1]) for l in lines]) + 3))
    L, R, Y = [], [], []
    for ln in lines:
        y = int(np.median(ln[:, 1]))
        xs = np.nonzero(tm[max(0, y - hh):y + hh + 1].any(0))[0]
        if len(xs) >= 10:
            L.append(xs.min()); R.append(xs.max()); Y.append(y)
    L, R, Y = (np.array(v, np.float32) for v in (L, R, Y))
    fl, fr = _edge_curve(L, Y), _edge_curve(R, Y)
    tol = max(2.5, 0.004 * g.shape[1])
    clean = [f is not None and f[1] <= tol and f[2] >= 0.6 * len(Y) for f in (fl, fr)]
    if not any(clean):
        # neither edge is a clean column edge (centred captions, a picture beside the text)
        info["reason"] = "no_clean_column_edges"
        return img, info
    ys = np.arange(g.shape[0], dtype=np.float32)
    y0, y1 = float(Y.min()), float(Y.max())
    yc = np.clip(ys, y0, y1)  # beyond the text, hold the edge where the text ends
    span = slice(int(y0), int(y1) + 1)
    if all(clean):  # justified: straighten both edges, evening out the width
        le, re = np.polyval(fl[0], yc), np.polyval(fr[0], yc)
        L0, R0 = float(np.median(le[span])), float(np.median(re[span]))
    else:  # ragged on one side: follow the clean edge, shifting rows without stretching
        x, f = (L, fl) if clean[0] else (R, fr)
        e = np.polyval(f[0], yc)
        e0 = float(np.median(e[span]))
        width = float(np.median(R - L))
        le, re = (e, e + width) if clean[0] else (e - width, e)
        L0, R0 = (e0, e0 + width) if clean[0] else (e0 - width, e0)
        info["edge"] = "left" if clean[0] else "right"
    width = re - le
    shift = max(np.abs(le - L0).max(), np.abs(re - R0).max())
    info["deshear_px"] = round(float(shift / s), 1)
    if shift < 1.0:
        info["reason"] = "upright"
        return img, info
    if shift > 0.10 * g.shape[1] or width.min() < 0.85 * (R0 - L0) or width.max() > 1.15 * (R0 - L0):
        info["reason"] = "implausible"
        info["deshear_px"] = 0.0
        return img, info
    # source x for each output x on each row: maps [L0, R0] onto that row's [le, re]
    scale_row = (width / (R0 - L0)).astype(np.float32)
    xs_out = np.arange(W0, dtype=np.float32) * s
    le_full = cv2.resize(le.reshape(-1, 1).astype(np.float32), (1, H0), interpolation=cv2.INTER_LINEAR).ravel()
    sc_full = cv2.resize(scale_row.reshape(-1, 1), (1, H0), interpolation=cv2.INTER_LINEAR).ravel()
    map_x = ((le_full[:, None] + (xs_out[None, :] - L0) * sc_full[:, None]) / s).astype(np.float32)
    map_y = np.repeat(np.arange(H0, dtype=np.float32)[:, None], W0, axis=1)
    out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, info
