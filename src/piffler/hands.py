"""Finger/hand detection and removal.

Old paper is often beige/yellow, which a plain skin-colour threshold confuses
with skin. We therefore require a pixel to be (a) skin-coloured in both YCrCb
and HSV, (b) clearly different from this frame's own paper colour, and
(c) part of a blob that reaches in from the edge of the region (hands enter
from outside the page) or is large.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def paper_color_lab(bgr: np.ndarray, region: Optional[np.ndarray] = None) -> np.ndarray:
    """Robust estimate of paper colour (Lab, float) from the brightest ~40% of pixels."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if region is not None:
        lab = lab[region.reshape(-1) > 0]
    if len(lab) < 50:
        return np.array([200, 128, 140], np.float32)
    L = lab[:, 0]
    sel = lab[L >= np.percentile(L, 60)]
    return np.median(sel, axis=0)


def skin_mask(bgr: np.ndarray, region: Optional[np.ndarray] = None, paper_delta: float = 14.0,
              min_area_frac: float = 0.0015, require_edge: bool = True) -> np.ndarray:
    """Return uint8 mask (255 = probable hand/finger)."""
    h, w = bgr.shape[:2]
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1].astype(np.int16), ycc[..., 2].astype(np.int16)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (Cr >= 136) & (Cr <= 180) & (Cb >= 77) & (Cb <= 130) & (Y > 40)
    m &= ((H <= 25) | (H >= 165)) & (S >= 35) & (S <= 210) & (V >= 50)

    # distance from paper colour (chroma weighted higher than lightness)
    paper = paper_color_lab(bgr, region)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.sqrt(0.25 * (lab[..., 0] - paper[0]) ** 2 + (lab[..., 1] - paper[1]) ** 2
                + (lab[..., 2] - paper[2]) ** 2)
    # skin is redder than paper (higher a*); yellowed paper is high b* but low a*
    a_, b_ = lab[..., 1] - 128.0, lab[..., 2] - 128.0
    hue = np.degrees(np.arctan2(b_, a_))
    # skin sits at Lab hue ~35-60 deg with clear redness; aged paper is ~65-90 deg (yellow)
    skinlike = (a_ >= 9) & (hue > 20) & (hue < 62) & ((lab[..., 1] - paper[1]) > 6)
    m &= (d > paper_delta) & skinlike

    mask = m.astype(np.uint8) * 255
    k = max(3, int(round(min(h, w) * 0.006)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * k, 3 * k)))

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, bool)
    min_area = min_area_frac * h * w
    edge = max(2, int(0.01 * min(h, w)))
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        if a < min_area:
            continue
        touches = x <= edge or y <= edge or x + bw >= w - edge or y + bh >= h - edge
        if touches or not require_edge or a > 0.02 * h * w:
            keep[i] = True
    out = (keep[lbl]).astype(np.uint8) * 255
    if region is not None:
        # a finger only matters where it overlaps the page; keep whole blob if it does
        n2, lbl2 = cv2.connectedComponents(out, connectivity=8)
        if n2 > 1:
            ov = np.bincount(lbl2[region > 0].ravel(), minlength=n2)
            out = ((ov > 0)[lbl2] & (lbl2 > 0)).astype(np.uint8) * 255
    return out


def skin_fraction(bgr_small: np.ndarray) -> float:
    try:
        return float((skin_mask(bgr_small) > 0).mean())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# multi-frame removal
# ---------------------------------------------------------------------------

def _gray_small(img, long_side=1000):
    s = long_side / max(img.shape[:2])
    s = min(1.0, s)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g), s


def align_to(ref: np.ndarray, img: np.ndarray, ref_mask: Optional[np.ndarray] = None,
             img_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Homography warping ``img`` onto ``ref`` (None if unreliable)."""
    g1, s1 = _gray_small(ref)
    g2, s2 = _gray_small(img)
    orb = cv2.ORB_create(3000)

    def feat_mask(m, g):
        if m is None:
            return None
        mm = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        return cv2.bitwise_not(cv2.dilate(mm, np.ones((9, 9), np.uint8)))

    k1, d1 = orb.detectAndCompute(g1, feat_mask(ref_mask, g1))
    k2, d2 = orb.detectAndCompute(g2, feat_mask(img_mask, g2))
    if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = bf.knnMatch(d2, d1, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 25:
        return None
    src = np.float32([k2[m.queryIdx].pt for m in good]) / s2
    dst = np.float32([k1[m.trainIdx].pt for m in good]) / s1
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0 / s1)
    if H is None or inl is None or inl.sum() < max(20, 0.3 * len(good)):
        return None
    # reject degenerate warps
    if abs(np.linalg.det(H[:2, :2])) < 0.5 or abs(np.linalg.det(H[:2, :2])) > 2.0:
        return None
    return H


def remove_fingers(ref: np.ndarray, others: List[np.ndarray], region: Optional[np.ndarray] = None,
                   paper_delta: float = 14.0) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Remove hands from ``ref`` using other frames of the same still page.

    Returns (clean_image, residual_mask, stats). Pixels that no other frame can
    fill are painted with the local paper background (good for margins); if
    that residual overlaps text, the caller should flag the page.
    """
    stats = {"finger_px_frac": 0.0, "filled_from_frames": 0.0, "residual_frac": 0.0, "aligned": 0}
    mask = skin_mask(ref, region, paper_delta)
    if region is not None:
        mask_in = cv2.bitwise_and(mask, region)
    else:
        mask_in = mask
    total = int((mask_in > 0).sum())
    if total == 0:
        return ref, np.zeros(ref.shape[:2], np.uint8), stats
    h, w = ref.shape[:2]
    stats["finger_px_frac"] = total / float(region.sum() / 255 if region is not None else h * w)
    pad = max(3, int(0.006 * max(h, w)))
    need = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)))
    out = ref.copy()
    remaining = need.copy()

    stack, valid = [], []
    for o in others:
        if o is None or o.shape != ref.shape:
            continue
        omask = skin_mask(o, None, paper_delta)
        H = align_to(ref, o, need, omask)
        if H is None:
            continue
        warped = cv2.warpPerspective(o, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        okm = cv2.warpPerspective(cv2.bitwise_not(cv2.dilate(omask, np.ones((2 * pad + 1,) * 2, np.uint8))),
                                  H, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        # also require photometric agreement outside the hand to guard against bad alignment
        stack.append(warped)
        valid.append(okm > 0)
        stats["aligned"] += 1

    if stack:
        V = np.stack(valid)
        cnt = V.sum(0)
        fillable = (remaining > 0) & (cnt > 0)
        if fillable.any():
            # per-pixel median over frames that are hand-free there (only touch needed pixels)
            Sf = np.stack([st[fillable] for st in stack]).astype(np.float32)   # (N, P, 3)
            Vf = V[:, fillable]
            med = np.nanmedian(np.where(Vf[..., None], Sf, np.nan), axis=0)
            # match donor brightness to the reference (exposure drifts between frames)
            ring = (cv2.dilate(need, np.ones((15, 15), np.uint8)) > 0) & (need == 0) & (cnt > 0)
            if ring.sum() > 50:
                Rf = np.stack([st[ring] for st in stack]).astype(np.float32)
                donor = np.nanmedian(np.where(V[:, ring][..., None], Rf, np.nan), axis=0)
                gain = (ref[ring].astype(np.float32).mean(0) + 1) / (np.nanmean(donor, 0) + 1)
                med = med * np.clip(gain, 0.7, 1.4)
            out[fillable] = np.clip(med, 0, 255).astype(np.uint8)
            remaining[fillable] = 0
            stats["filled_from_frames"] = float(fillable.sum()) / max(1, (need > 0).sum())

    residual = remaining
    if residual.any():
        out = fill_with_paper(out, residual)
    if region is not None:
        res_in = cv2.bitwise_and(residual, region)
        stats["residual_frac"] = float((res_in > 0).sum()) / float(max(1, region.sum() / 255))
    else:
        stats["residual_frac"] = float((residual > 0).mean())
    return out, residual, stats


def fill_with_paper(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked area with a smooth estimate of the surrounding paper.

    Large-scale background is estimated from unmasked pixels (text removed by
    a max filter) and diffused into the hole; small holes get Telea inpainting.
    """
    if not mask.any():
        return img
    h, w = img.shape[:2]
    s = 400.0 / max(h, w)
    s = min(1.0, s)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    # remove text (dark) by dilation = local max, giving paper background
    bg = cv2.dilate(small, np.ones((7, 7), np.uint8))
    msk = cv2.dilate(msmall, np.ones((5, 5), np.uint8))
    bg = cv2.inpaint(bg, msk, 9, cv2.INPAINT_TELEA)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    out = img.copy()
    # feather the seam
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.002 * max(h, w)))
    soft = np.maximum(soft, (mask > 0).astype(np.float32))[..., None]
    out = (out.astype(np.float32) * (1 - soft) + bg.astype(np.float32) * soft).astype(np.uint8)
    return out
