"""Turn a flattened page photo into a clean, high-contrast "scan".

The page photo still shows a bit of countertop at its edges, uneven paper
tone, pale traces of fingers that were painted over, and grey-on-grey text.
This keeps the print (text and pictures), makes the paper white and the ink
dark, and drops marks that aren't print.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _background(gray: np.ndarray) -> np.ndarray:
    """Paper brightness everywhere, with the print removed."""
    h, w = gray.shape
    s = min(1.0, 800.0 / max(h, w))
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    k = max(7, int(0.025 * max(g.shape)) | 1)  # wider than a text stroke, narrower than shading
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg, min(k, 255))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _paper_mask(gray: np.ndarray) -> np.ndarray:
    """The sheet itself: the paper-bright region joined to the page centre, so
    countertop and the facing page's edge fall outside. Judged on a median
    (not a closing, which would let a speckled counter's bright flecks spread)."""
    h, w = gray.shape
    s = min(1.0, 500.0 / max(h, w))
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    k = max(5, int(0.03 * max(g.shape)) | 1)
    med = cv2.medianBlur(g, min(k, 255))
    lvl = float(np.median(med[g.shape[0] // 4: 3 * g.shape[0] // 4, g.shape[1] // 4: 3 * g.shape[1] // 4]))
    m = (med > 0.82 * lvl).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        c = lbl[g.shape[0] // 2, g.shape[1] // 2] or 1 + int(np.argmax(st[1:, 4]))
        m = (lbl == c).astype(np.uint8)
        # pictures and big dark print are holes in the paper: fill them
        ff = np.pad(m, 1)
        cv2.floodFill(ff, None, (0, 0), 1, flags=4 | (2 << 8))
        m = m | (ff[1:-1, 1:-1] == 0).astype(np.uint8)
    m = cv2.erode(m, np.ones((3, 3), np.uint8))
    return cv2.resize(m * 255, (w, h), interpolation=cv2.INTER_LINEAR) > 127


def _pictures(norm: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """Photos and drawings: areas that stay dark on average over a window much
    larger than a line of text."""
    h, w = norm.shape
    k = max(15, int(0.04 * max(h, w)) | 1)
    local = cv2.blur(norm, (k, k))
    pic = ((local < 0.75) & paper).astype(np.uint8)
    pic = cv2.morphologyEx(pic, cv2.MORPH_OPEN, np.ones((k // 2 | 1,) * 2, np.uint8))
    return cv2.dilate(pic, np.ones((k, k), np.uint8)) > 0


def _stroke_level(norm: np.ndarray, paper: np.ndarray, pics: np.ndarray) -> float:
    """How dark this page's text strokes really are (thin print blurs to grey)."""
    cores = cv2.erode(norm, np.ones((5, 5), np.uint8))
    v = cores[paper & ~pics & (cores < 0.9)]
    return float(np.percentile(v, 30)) if v.size > 200 else 0.5


def _faint_marks(tone: np.ndarray, paper: np.ndarray, pics: np.ndarray) -> np.ndarray:
    """Large marks that never get properly dark: finger ghosts, shadows and
    smudges. Small faint marks are kept: they are light print (italics, captions)."""
    h, w = tone.shape
    ink = ((tone < 0.85) & paper & ~pics).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    if n <= 1:
        return np.zeros(tone.shape, bool)
    darkest = np.ones(n, np.float32)
    np.minimum.at(darkest, lbl.ravel(), tone.ravel())
    big = (st[:, 3] > 0.06 * h) | (st[:, 2] > 0.25 * w)  # taller than a few lines, or wider than a word run
    faint = (darkest > 0.45) & big
    faint[0] = False
    return faint[lbl]


def _edge_junk(tone: np.ndarray, paper: np.ndarray, pics: np.ndarray, gutter: Optional[str]) -> np.ndarray:
    """Ink touching the outer edge of the page or the paper's outline: countertop,
    the stacked edges of other pages, a thumb. Print keeps a margin, except
    at the gutter, where a pocket book's text can run to the binding."""
    h, w = tone.shape
    ink = (tone < 0.85).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    if n <= 1:
        return np.zeros(tone.shape, bool)
    e = max(2, int(0.004 * max(h, w)))
    border = np.zeros((h, w), bool)
    border[:e, :] = border[-e:, :] = True
    if gutter != "left":
        border[:, :e] = True
    if gutter != "right":
        border[:, -e:] = True
    off = cv2.dilate((~paper).astype(np.uint8), np.ones((2 * e + 1,) * 2, np.uint8)) > 0
    if gutter == "right":
        off[:, int(0.9 * w):] = False
    elif gutter == "left":
        off[:, :int(0.1 * w)] = False
    touch = np.zeros(n, bool)
    touch[np.unique(lbl[(border | off) & (ink > 0)])] = True
    touch[0] = False
    if pics.any():
        touch[np.unique(lbl[pics & (ink > 0)])] = False
    return touch[lbl]


def clean_scan(img: np.ndarray, color: Optional[bool] = None, upscale: float = 2.0,
               gutter: Optional[str] = None, overlap: int = 0) -> np.ndarray:
    """Flattened page photo (BGR) -> clean, high-contrast scan, ``upscale`` times
    larger (gray, or BGR for pages with colour print). ``gutter`` says which
    side ("left"/"right") is the binding, if the page came from a spread, and
    ``overlap`` how many pixels past the binding the page was cut: marks lying
    wholly in that strip are the facing page's."""
    if upscale and upscale != 1:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = lab[..., 0].astype(np.float32)
    bg = _background(lab[..., 0])
    paper = _paper_mask(lab[..., 0])
    norm = np.clip(gray / np.maximum(bg, 20), 0, 1.2)
    # sharpen: recovers some of the stroke contrast lost to optics and compression
    sig = max(1.0, 0.6 * (upscale or 1))
    norm = norm + 0.9 * (norm - cv2.GaussianBlur(norm, (0, 0), sig))
    # levels: paper to white, the typical stroke core to near-black
    pics = _pictures(norm, paper)
    black = _stroke_level(norm, paper, pics) - 0.08
    white = 0.94
    tone = np.clip((norm - black) / max(0.08, white - black), 0, 1)
    if gutter in ("left", "right") and pics.any():
        # a narrow picture against the binding is the facing page's, cut by the split
        n, lbl, st, _ = cv2.connectedComponentsWithStats(pics.astype(np.uint8), 8)
        W = pics.shape[1]
        for i in range(1, n):
            x, ww = st[i, 0], st[i, 2]
            at_gutter = x + ww >= W - 2 if gutter == "right" else x <= 1
            if at_gutter and ww < 0.15 * W:
                tone[lbl == i] = 1.0
                pics[lbl == i] = False
    if gutter in ("left", "right"):
        # a dense, tall block against the binding is part of the facing page's
        # picture; text running to the binding is never that solid
        Hh, W = tone.shape
        dense = cv2.morphologyEx((tone < 0.6).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lbl, st, _ = cv2.connectedComponentsWithStats(dense, 8)
        for i in range(1, n):
            x, y, ww, hh, a = st[i]
            at_gutter = x + ww >= W - 3 if gutter == "right" else x <= 2
            if at_gutter and hh > 0.08 * Hh and a > 0.5 * ww * hh and ww < 0.2 * W:
                tone[y:y + hh, x:x + ww][lbl[y:y + hh, x:x + ww] == i] = 1.0
    if gutter in ("left", "right") and overlap > 0:
        band = int(round(overlap * (upscale or 1)))  # just the part past the split line
        n, lbl, st, _ = cv2.connectedComponentsWithStats((tone < 0.85).astype(np.uint8), 8)
        W = tone.shape[1]
        x0, x1 = st[:, 0], st[:, 0] + st[:, 2]
        inside = (x0 >= W - band) if gutter == "right" else (x1 <= band)
        inside[0] = False
        tone[inside[lbl]] = 1.0
    tone[_faint_marks(tone, paper, pics)] = 1.0
    if pics.any():  # pictures get gentle levels: full text contrast crushes their shadows
        soft = np.clip((norm - 0.04) / 0.92, 0, 1)
        tone[pics] = soft[pics]
    tone[_edge_junk(tone, paper, pics, gutter)] = 1.0
    tone[~paper] = 1.0
    out8 = (tone * 255 + 0.5).astype(np.uint8)
    if color is None:  # colour only where the print itself is coloured
        chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
        ink = tone < 0.5
        color = bool(ink.sum() > 500 and np.percentile(chroma[ink], 90) > 18)
    if not color:
        return out8
    lab2 = lab.copy()
    lab2[..., 0] = out8
    lab2[..., 1:][tone > 0.9] = 128  # paper stays neutral white
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
