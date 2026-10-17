"""Find the moments where a page is held still, and pick the best frames.

Pass 1 streams the video at ``analysis_fps`` keeping only scalar features
(memory stays flat for long videos). Still segments are found with a
threshold adapted to each video's own motion distribution, so handheld
shake, tripods, and slow-mo footage all work without tuning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

from .config import Config
from .hands import skin_fraction
from .video import VideoSource

log = logging.getLogger(__name__)


@dataclass
class FrameStats:
    idx: np.ndarray
    t: np.ndarray
    motion: np.ndarray
    sharp: np.ndarray
    bright: np.ndarray
    clip_hi: np.ndarray
    skin: np.ndarray


@dataclass
class Segment:
    start: int          # frame index
    end: int
    t0: float
    t1: float
    candidates: List[int]
    scores: List[float]
    flags: List[str] = field(default_factory=list)

    @property
    def best(self) -> int:
        return self.candidates[int(np.argmax(self.scores))]


def analyze(src: VideoSource, cfg: Config, progress=None) -> FrameStats:
    fps = src.info.fps
    step = max(1, int(round(fps / max(1.0, cfg.analysis_fps))))
    idxs, motion, sharp, bright, clip, skin = [], [], [], [], [], []
    prev = None
    aw = cfg.analysis_width
    for i, frame in src.frames(step, width=2 * aw):  # all analysis needs; ffmpeg scales on the GPU
        h, w = frame.shape[:2]
        s = aw / float(w)
        small = cv2.resize(frame, (aw, max(8, int(round(h * s)))), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gb = cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)
        gn = gb / (gb.mean() + 1.0)  # exposure-normalised
        motion.append(np.nan if prev is None else float(np.mean(np.abs(gn - prev))) * 100.0)
        prev = gn
        # sharpness on a 2x larger central crop (text detail), normalised by contrast
        s2 = min(1.0, 2 * aw / float(w))
        mid = cv2.resize(frame, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA)
        mg = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
        H, W = mg.shape
        c = mg[H // 6: H - H // 6, W // 6: W - W // 6]
        lap = cv2.Laplacian(c, cv2.CV_32F)
        sharp.append(float(lap.var()) / (float(c.std()) + 5.0))
        bright.append(float(g.mean()))
        clip.append(float((g >= 250).mean()))
        skin.append(skin_fraction(small))
        idxs.append(i)
        if progress and len(idxs) % 50 == 0:
            progress(i)
    if not idxs:
        raise RuntimeError("No decodable frames")
    m = np.array(motion, np.float32)
    if len(m) > 1:
        m[0] = m[1]
    else:
        m[:] = 0
    return FrameStats(np.array(idxs), np.array(idxs, np.float64) / fps, m, np.array(sharp, np.float32),
                      np.array(bright, np.float32), np.array(clip, np.float32), np.array(skin, np.float32))


def _runs(mask: np.ndarray):
    """(start, end_exclusive) of True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _quality(st: FrameStats) -> np.ndarray:
    sharp = st.sharp / (np.percentile(st.sharp, 90) + 1e-6)
    b = st.bright
    expo = np.clip(1.0 - np.abs(b - 150) / 150.0, 0.2, 1.0) * np.clip(1.0 - 4 * st.clip_hi, 0.1, 1.0)
    hand = np.clip(1.0 - 3.0 * st.skin, 0.2, 1.0)
    return sharp * expo * hand


def find_segments(st: FrameStats, cfg: Config, fps: float) -> List[Segment]:
    n = len(st.idx)
    q = _quality(st)
    if n < 3:
        k = int(np.argmax(q))
        return [Segment(int(st.idx[0]), int(st.idx[-1]), float(st.t[0]), float(st.t[-1]),
                        [int(st.idx[k])], [float(q[k])], ["very_short_video"])]

    m = st.motion.copy()
    # median-3 smoothing kills single-frame spikes (compression glitches)
    m = np.median(np.stack([np.r_[m[0], m[:-1]], m, np.r_[m[1:], m[-1]]]), axis=0)
    dt = float(np.median(np.diff(st.t))) if n > 1 else 1 / fps
    lo, hi = np.percentile(m, 20), np.percentile(m, 90)

    segs = []
    k = cfg.motion_k
    min_hold = cfg.min_hold_s
    for attempt in range(4):
        thr = lo + k * (hi - lo) + 1e-3
        still = m <= thr
        # bridge one-sample dropouts inside a hold
        for a, b in _runs(~still):
            if b - a == 1 and a > 0 and b < n:
                still[a:b] = True
        runs = [(a, b) for a, b in _runs(still) if (st.t[b - 1] - st.t[a] + dt) >= min_hold * 0.5]
        if runs:
            break
        k *= 1.8
        min_hold *= 0.7
        log.info("No still segments found; relaxing threshold (attempt %d)", attempt + 1)

    if not runs:
        # last resort: sharpness peaks at least 1 s apart
        log.warning("Could not find still segments; falling back to sharpness peaks")
        order = np.argsort(-q)
        chosen = []
        for i in order:
            if all(abs(st.t[i] - st.t[j]) > 1.0 for j in chosen):
                chosen.append(i)
        chosen.sort()
        return [Segment(int(st.idx[i]), int(st.idx[i]), float(st.t[i]), float(st.t[i]),
                        [int(st.idx[i])], [float(q[i])], ["no_still_segments"]) for i in chosen]

    for a, b in runs:
        dur = st.t[b - 1] - st.t[a] + dt
        flags = []
        if dur < min_hold:
            flags.append("short_hold")
        # trim the edges of the hold (residual motion blur as the hand leaves/arrives)
        L = b - a
        trim = int(L * 0.15) if L >= 6 else 0
        a2, b2 = a + trim, b - trim
        idx = np.arange(a2, b2)
        qq = q[idx]
        # choose up to K candidates, spread out in time (fingers move -> better compositing)
        K = max(1, cfg.candidates_per_page)
        gap = max(1, len(idx) // (2 * K))
        chosen = []
        for j in np.argsort(-qq):
            if all(abs(j - c) >= gap for c in chosen):
                chosen.append(j)
            if len(chosen) >= K:
                break
        chosen.sort(key=lambda j: -qq[j])
        segs.append(Segment(int(st.idx[a]), int(st.idx[b - 1]), float(st.t[a]), float(st.t[b - 1]),
                            [int(st.idx[idx[j]]) for j in chosen], [float(qq[j]) for j in chosen], flags))

    # global quality flags
    med_sharp = np.median([max(s.scores) for s in segs]) if segs else 0
    for s in segs:
        if max(s.scores) < 0.45 * med_sharp:
            s.flags.append("low_quality_frame")
    return segs
