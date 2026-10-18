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


def picture_mask(gray: np.ndarray, gutter: Optional[str] = None) -> np.ndarray:
    """Photos and drawings, from raw brightness: areas that stay well below the
    paper's level on average over a window much larger than a line of text.
    (Judged before any lighting correction, which would flatten them.)"""
    h, w = gray.shape
    s = min(1.0, 500.0 / max(h, w))
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
    k = max(9, int(0.05 * max(g.shape)) | 1)
    paper = float(np.percentile(g, 85))
    # shading (gutter, lamp falloff) is smooth; compare with a much wider view
    wide = cv2.GaussianBlur(cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((3 * k, 3 * k), np.uint8)), (0, 0), k)
    local = cv2.blur(g, (k, k))
    m = ((local < 0.72 * np.minimum(wide, paper)) | (local < 0.6 * paper)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k // 2 | 1,) * 2, np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, 4] > 0.004 * g.size  # bigger than a heading's worth of dark
    # and full of detail: a smooth dark wedge (shadow, countertop past the page) isn't a picture
    detail = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F))
    mean_detail = np.bincount(lbl.ravel(), weights=detail.ravel(), minlength=n) / np.maximum(st[:, 4], 1)
    keep &= mean_detail > 2.5
    # printed pictures sit inside the margins; dark stuff at the page border is
    # countertop, a hand or the next page (the gutter side is handled by the caller)
    e = 2
    x, y, bw, bh = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    gh, gw = g.shape
    keep &= ~((y <= e) | (y + bh >= gh - e))
    keep &= ~((x <= e) & (x + bw >= gw - e))  # spanning the whole width: not a print
    if gutter == "right":  # the outer edge is the left one
        keep &= ~(x <= e)
    elif gutter == "left":
        keep &= ~(x + bw >= gw - e)
    m = np.zeros_like(m)
    for i in np.nonzero(keep)[0]:  # a photo's bright parts (a white shirt) are still the photo
        cnts, _ = cv2.findContours((lbl == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(m, [cv2.convexHull(np.vstack(cnts))], 1)
    m = cv2.dilate(m, np.ones((k // 3 | 1,) * 2, np.uint8))
    return cv2.resize(m * 255, (w, h), interpolation=cv2.INTER_NEAREST) > 0


def paper_background(img: np.ndarray, pics: Optional[np.ndarray] = None, long_side: int = 800) -> np.ndarray:
    """Paper brightness everywhere (gray or BGR), with print removed and, over
    pictures, filled in from the paper around them: a picture is not shading."""
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(h, w))
    g = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    k = max(7, int(0.025 * max(g.shape[:2])) | 1)  # wider than a stroke, narrower than shading
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg, min(k, 255))
    if pics is not None and pics.any():
        pm = cv2.resize(pics.astype(np.uint8) * 255, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        pm = cv2.dilate(pm, np.ones((k, k), np.uint8))
        bg = cv2.inpaint(bg, pm, max(3, k), cv2.INPAINT_TELEA)
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
    lvl = float(np.percentile(med, 90))  # paper is the bright part (the centre may be a photo)
    m = (med > 0.82 * lvl).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        c = 1 + int(np.argmax(st[1:, 4]))
        m = (lbl == c).astype(np.uint8)
        # pictures and big dark print are holes in the paper: fill them
        ff = np.pad(m, 1)
        cv2.floodFill(ff, None, (0, 0), 1, flags=4 | (2 << 8))
        m = m | (ff[1:-1, 1:-1] == 0).astype(np.uint8)
    m = cv2.erode(m, np.ones((3, 3), np.uint8))
    return cv2.resize(m * 255, (w, h), interpolation=cv2.INTER_LINEAR) > 127


def _stroke_width(ink: np.ndarray) -> float:
    """Typical stroke width of the print, in pixels (2x the distance to the
    nearest paper, taken at stroke centres)."""
    if ink.sum() < 2e-4 * ink.size:
        return 2.0
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt > 0)
    return float(np.clip(2.0 * np.median(dt[ridge]), 1.0, 40.0)) if ridge.any() else 2.0


def _stroke_level(norm: np.ndarray, paper: np.ndarray, pics: np.ndarray) -> float:
    """How dark this page's text strokes really are (thin print blurs to grey)."""
    cores = cv2.erode(norm, np.ones((5, 5), np.uint8))
    v = cores[paper & ~pics & (cores < 0.9)]
    return float(np.percentile(v, 30)) if v.size > 2e-4 * norm.size else 0.5


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


def _picture_tones(norm: np.ndarray, pics: np.ndarray) -> np.ndarray:
    """Photos: their own levels (darkest shadow to brightest highlight), gentle
    noise reduction, no text sharpening. 1.0 = paper white."""
    out = np.ones_like(norm)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(pics.astype(np.uint8), 8)
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        m = lbl[y:y + h, x:x + w] == i
        v = norm[y:y + h, x:x + w]
        g8 = (np.clip(v, 0, 1) * 255).astype(np.uint8)
        g8 = cv2.fastNlMeansDenoising(g8, None, h=7, templateWindowSize=7, searchWindowSize=21)
        v = g8.astype(np.float32) / 255
        lo, hi = np.percentile(v[m], [0.5, 99.7])
        stretched = np.clip((v - lo) / max(0.05, hi - lo), 0, 1)
        v = 0.75 * stretched + 0.25 * np.clip(v, 0, 1)  # full range, without crushing shadows
        v = 0.04 + 0.96 * v
        o = out[y:y + h, x:x + w]
        o[m] = v[m]
    return out


def _faint_outside_text(tone: np.ndarray, paper: np.ndarray, pics: np.ndarray,
                        boxes: Optional[list] = None) -> np.ndarray:
    """Faint marks away from any text (the traces a hand leaves in the margin).
    Text is where OCR read words (``boxes``, in this image's pixels), else the
    block spanned by dark print; nothing printed elsewhere is faint."""
    h, w = tone.shape
    block = np.zeros(tone.shape, np.uint8)
    lh = 0.0
    if boxes:
        for b in boxes:
            cv2.fillPoly(block, [np.asarray(b, np.int32).reshape(-1, 2)], 1)
        heights = [np.ptp(np.asarray(b)[:, 1]) for b in boxes]
        lh = max(3.0, float(np.median(heights)))
        # half a line above and below (accents, descenders); sideways barely
        # past the word, since hands leave their traces just beside line starts
        ry, rx = int(0.5 * lh), int(0.25 * lh)
        block = cv2.dilate(block, np.ones((2 * ry + 1, 2 * rx + 1), np.uint8))
    else:
        ys, xs = np.nonzero((tone < 0.35) & paper & ~pics)
        if len(xs) < 2e-4 * tone.size:
            return np.zeros(tone.shape, bool)
        x0, x1 = np.percentile(xs, [0.5, 99.5])
        y0, y1 = np.percentile(ys, [0.5, 99.5])
        pad = 0.015 * max(h, w)
        block[int(max(0, y0 - pad)):int(y1 + pad) + 1, int(max(0, x0 - pad)):int(x1 + pad) + 1] = 1
    block = (block > 0) | pics
    ink = ((tone < 0.9) & paper).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    if n <= 1:
        return np.zeros(tone.shape, bool)
    darkest = np.ones(n, np.float32)
    np.minimum.at(darkest, lbl.ravel(), tone.ravel())
    touches = np.zeros(n, bool)
    touches[np.unique(lbl[block & (ink > 0)])] = True
    out = (darkest > 0.4) & ~touches
    if boxes:  # also tall, thin strokes away from all words: the edge of a hand's shadow
        out |= ~touches & (st[:, 3] > 2.5 * lh) & (st[:, 2] < 0.6 * lh)
    out[0] = False
    return out[lbl]


def clean_scan(img: np.ndarray, color: Optional[bool] = None, upscale: float = 2.0,
               gutter: Optional[str] = None, overlap: int = 0, style: str = "natural",
               word_boxes: Optional[list] = None) -> np.ndarray:
    """Flattened page photo (BGR) -> clean scan, ``upscale`` times larger.

    ``style`` "natural" keeps the look of the page: paper in the book's own
    (brightened) tint, dark but soft-edged text, photos in their own tones,
    BGR out. "bw" is a crisp black-on-white scan, gray out (BGR only for
    pages with coloured print). ``gutter`` says which side ("left"/"right")
    is the binding, if the page came from a spread, and ``overlap`` how many
    pixels past the binding the page was cut: marks lying wholly in that
    strip are the facing page's."""
    bw = style == "bw"
    up = upscale or 1
    if upscale and upscale != 1:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = lab[..., 0].astype(np.float32)
    pics = picture_mask(lab[..., 0], gutter)
    paper = _paper_mask(lab[..., 0])
    if pics.any() and paper.any():
        # pictures are part of the sheet, but only within its outline: a dark
        # wedge of countertop beyond the page edge is not a picture
        hull = np.zeros(paper.shape, np.uint8)
        cnts, _ = cv2.findContours(paper.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(hull, [cv2.convexHull(np.vstack(cnts))], 1)
        pics &= hull > 0
        paper |= pics
    bg = paper_background(lab[..., 0], pics)
    norm = np.clip(gray / np.maximum(bg, 20), 0, 1.2)
    # the page's own text scale: line height from OCR boxes, else from the print
    if word_boxes:
        line_h = up * float(np.median([np.ptp(np.asarray(b)[:, 1]) for b in word_boxes]))
    else:
        line_h = 8.0 * _stroke_width(norm < 0.7)
    # sharpen print (not pictures): recovers stroke contrast lost to optics and compression
    # sharpen at the scale of the strokes themselves
    sig = max(1.0, 0.5 * _stroke_width(norm < 0.7))
    sharp = norm + (0.9 if bw else 0.6) * (norm - cv2.GaussianBlur(norm, (0, 0), sig))
    # levels: paper to white, the typical stroke core to near-black
    black = _stroke_level(sharp, paper, pics) - (0.08 if bw else 0.05)
    white = 0.94 if bw else 0.975
    tone = np.clip((sharp - black) / max(0.08, white - black), 0, 1)
    if not bw:
        tone = tone ** 0.9  # keep the soft edges of strokes
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
        kd = max(5, int(0.6 * line_h) | 1)  # closes the gaps between words, not between lines
        dense = cv2.morphologyEx((tone < 0.6).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kd, kd), np.uint8))
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
    boxes = [np.asarray(b, np.float32) * up for b in word_boxes] if word_boxes else None
    tone[_faint_outside_text(tone, paper, pics, boxes)] = 1.0
    if pics.any():
        tone[pics] = _picture_tones(norm, pics)[pics]
    tone[_edge_junk(tone, paper, pics, gutter)] = 1.0
    tone[~paper] = 1.0
    if bw:
        out8 = (tone * 255 + 0.5).astype(np.uint8)
        if color is None:  # colour only where the print itself is coloured
            chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
            ink = tone < 0.5
            color = bool(ink.sum() > 2e-4 * ink.size and np.percentile(chroma[ink], 90) > 18)
        if not color:
            return out8
        lab2 = lab.copy()
        lab2[..., 0] = out8
        lab2[..., 1:][tone > 0.9] = 128  # paper stays neutral white
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    # natural: the page's own paper colour, lightened as a good scanner would
    sheet = paper & ~pics & (tone > 0.97)
    pl = np.median(lab[sheet].reshape(-1, 3), axis=0).astype(np.float32) if sheet.sum() > 2e-4 * sheet.size         else np.array([240, 128, 130], np.float32)
    paper_L = float(np.clip(pl[0] * 1.05, 232, 246))
    tint = np.clip(pl[1:] - 128, -6, 8)  # keep a hint of the paper's colour, never a cast
    ink_L = 22.0
    L = ink_L + (paper_L - ink_L) * tone
    ab = np.empty(lab.shape[:2] + (2,), np.float32)
    ab[:] = tint * np.clip(tone, 0, 1)[..., None]  # ink neutral, paper tinted
    if pics.any():  # photos keep their own colour, a little calmer
        ab[pics] = (lab[..., 1:][pics].astype(np.float32) - 128) * 0.85
        L[pics] = paper_L * tone[pics]
    lab2 = np.dstack([np.clip(L, 0, 255), np.clip(ab + 128, 0, 255)]).astype(np.uint8)
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
