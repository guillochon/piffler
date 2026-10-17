"""Robust frame access for phone videos.

Handles: missing/zero/NaN fps, wrong frame counts, truncated or partially
corrupt files, rotation, and files OpenCV can't open directly (transcoded
via ffmpeg if it is installed).
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterator, Iterable, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count_hint: int  # container's claim; often wrong, never trusted for logic


_ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _open(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    try:  # honour phone rotation metadata where the backend supports it
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass
    return cap


def _probe_ok(cap: cv2.VideoCapture) -> bool:
    if not cap.isOpened():
        return False
    ok, frame = cap.read()
    return bool(ok) and frame is not None and frame.size > 0


def _transcode(path: str) -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    fd, out = tempfile.mkstemp(suffix=".mp4", prefix="piffler_")
    os.close(fd)
    cmd = [ffmpeg, "-v", "error", "-y", "-err_detect", "ignore_err", "-i", path,
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", out]
    log.warning("OpenCV could not decode %s; transcoding with ffmpeg", path)
    try:
        subprocess.run(cmd, check=False, timeout=3600, capture_output=True)
    except Exception as e:  # pragma: no cover
        log.error("ffmpeg failed: %s", e)
        return None
    return out if os.path.exists(out) and os.path.getsize(out) > 0 else None


class VideoSource:
    """Sequential, fault-tolerant reader. Use ``frames()`` or ``fetch()``."""

    def __init__(self, path: str, rotate: int = 0, max_bad_reads: int = 30):
        if not os.path.exists(path):
            raise VideoError(f"File not found: {path}")
        if os.path.getsize(path) == 0:
            raise VideoError(f"File is empty: {path}")
        self.rotate = int(rotate) % 360
        if self.rotate not in _ROT:
            raise VideoError("rotate must be one of 0, 90, 180, 270")
        self.max_bad_reads = max_bad_reads
        self._tmp = None
        self.path = path

        cap = _open(path)
        if not _probe_ok(cap):
            cap.release()
            alt = _transcode(path)
            if alt is None:
                raise VideoError(f"Could not decode any frames from {path}")
            self._tmp = alt
            self.path = alt
            cap = _open(alt)
            if not _probe_ok(cap):
                cap.release()
                raise VideoError(f"Could not decode any frames from {path} (even after transcoding)")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or not math.isfinite(fps) or fps <= 1 or fps > 1000:
            log.warning("Unreliable fps (%r); assuming 30", fps)
            fps = 30.0
        ok, f0 = True, None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, f0 = cap.read()
        if not ok or f0 is None:  # some backends can't seek back; reopen
            cap.release()
            cap = _open(self.path)
            ok, f0 = cap.read()
        f0 = self._prep(f0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        self.info = VideoInfo(path, float(fps), f0.shape[1], f0.shape[0], max(n, 0))

    # -- helpers ----------------------------------------------------------
    def _prep(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.dtype != np.uint8:  # e.g. 16-bit decodes
            frame = cv2.convertScaleAbs(frame, alpha=255.0 / max(1, frame.max()))
        r = _ROT[self.rotate]
        return cv2.rotate(frame, r) if r is not None else frame

    def frames(self, step: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_index, BGR frame) for every ``step``-th decodable frame.

        Undecodable frames are skipped; a long run of failures is treated as
        the end of a truncated file rather than an error.
        """
        step = max(1, int(step))
        cap = _open(self.path)
        idx, bad = -1, 0
        try:
            while True:
                idx += 1
                if idx % step:
                    if not cap.grab():
                        bad += 1
                        if bad > self.max_bad_reads:
                            break
                    else:
                        bad = 0
                    continue
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    bad += 1
                    if bad > self.max_bad_reads:
                        break
                    continue
                bad = 0
                try:
                    yield idx, self._prep(frame)
                except Exception as e:  # malformed frame
                    log.debug("skipping frame %d: %s", idx, e)
        finally:
            cap.release()

    def iter_fetch(self, indices: Iterable[int]) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (index, frame) for the requested frames in one sequential pass.

        Sequential grab() is slower than seeking but exact; seeking is
        unreliable on variable-frame-rate phone video. Frames that can't be
        decoded are silently missing from the output.
        """
        wanted = sorted(set(int(i) for i in indices))
        if not wanted:
            return
        want = set(wanted)
        last = wanted[-1]
        cap = _open(self.path)
        idx, bad = -1, 0
        try:
            while idx < last:
                idx += 1
                if idx in want:
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size:
                        bad = 0
                        try:
                            yield idx, self._prep(frame)
                        except Exception as e:
                            log.debug("bad frame %d: %s", idx, e)
                        continue
                elif cap.grab():
                    bad = 0
                    continue
                bad += 1
                if bad > self.max_bad_reads:
                    break
        finally:
            cap.release()

    def fetch(self, indices: Iterable[int]) -> dict:
        return dict(self.iter_fetch(indices))

    def close(self):
        if self._tmp and os.path.exists(self._tmp):
            try:
                os.remove(self._tmp)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
