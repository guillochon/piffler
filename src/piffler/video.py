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


class _FFmpeg:
    """Frames straight from ffmpeg: multi-threaded, on the GPU (NVDEC) when it can.

    OpenCV decodes on one thread and hands over full frames, so a 4K phone
    video spends most of its time decoding and shrinking pixels that analysis
    throws away. ffmpeg picks frames and scales them before they reach Python.
    """

    _cuda_ok: dict = {}

    def __init__(self, path: str, use_gpu: bool = True):
        self.exe = shutil.which("ffmpeg")
        probe = shutil.which("ffprobe")
        if not self.exe or not probe:
            raise VideoError("ffmpeg not found")
        import json
        try:
            out = subprocess.run([probe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", path],
                                 capture_output=True, text=True, timeout=60).stdout
            st = json.loads(out)["streams"][0]
        except Exception as e:
            raise VideoError(f"ffprobe failed: {e}")
        self.path = path
        self.w, self.h = int(st["width"]), int(st["height"])
        # clockwise turn needed for display: the display matrix states it
        # counter-clockwise, the older "rotate" tag clockwise
        self.cw = int(st.get("tags", {}).get("rotate", 0) or 0) % 360
        for sd in st.get("side_data_list", []):
            if sd.get("rotation") is not None:
                self.cw = (-int(round(float(sd["rotation"])))) % 360
        self.cw = self.cw if self.cw in _ROT else 0
        self.gpu = use_gpu and self._has_cuda()

    def _has_cuda(self) -> bool:
        if self.path not in _FFmpeg._cuda_ok:
            cmd = [self.exe, "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", self.path,
                   "-frames:v", "1", "-vf", "scale_cuda=64:64:format=yuv420p,hwdownload,format=yuv420p",
                   "-f", "null", "-"]
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=60)
                _FFmpeg._cuda_ok[self.path] = r.returncode == 0
            except Exception:
                _FFmpeg._cuda_ok[self.path] = False
            log.info("GPU video decoding: %s", "on" if _FFmpeg._cuda_ok[self.path] else "off")
        return _FFmpeg._cuda_ok[self.path]

    def read(self, select: str, width: Optional[int] = None, extra_cw: int = 0, limit: Optional[int] = None):
        """Yield frames passing ffmpeg's `select` expression, `width` pixels wide
        after rotation (full size when None), as uint8 BGR."""
        turn = (self.cw + extra_cw) % 360
        sw, sh = (self.w, self.h)
        if width:
            # target size before rotation; even numbers keep the scaler happy
            if turn in (90, 270):
                sh = int(width) // 2 * 2
                sw = int(round(self.w * sh / self.h)) // 2 * 2
            else:
                sw = int(width) // 2 * 2
                sh = int(round(self.h * sw / self.w)) // 2 * 2
        if self.gpu:
            pre = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            vf = f"select='{select}',scale_cuda={sw}:{sh}:format=yuv420p,hwdownload,format=yuv420p"
        else:
            pre = ["-threads", "0"]
            vf = f"select='{select}'" + (f",scale={sw}:{sh}:flags=area" if width else "")
        cmd = [self.exe, "-v", "error", "-nostdin", "-noautorotate", *pre, "-i", self.path, "-an", "-vf", vf,
               "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo"]
        if limit:
            cmd += ["-frames:v", str(limit)]
        p = subprocess.Popen(cmd + ["-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        size = sw * sh * 3
        r = _ROT[turn]
        try:
            while True:
                buf = p.stdout.read(size)
                if len(buf) < size:
                    break
                f = np.frombuffer(buf, np.uint8).reshape(sh, sw, 3)
                yield cv2.rotate(f, r) if r is not None else f.copy()
        finally:
            p.stdout.close()
            p.kill()
            p.wait()


class VideoSource:
    """Sequential, fault-tolerant reader. Use ``frames()`` or ``fetch()``."""

    def __init__(self, path: str, rotate: int = 0, max_bad_reads: int = 30, use_gpu: bool = True):
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
        self._ff = self._try_ffmpeg(f0, use_gpu)
        self.decoder = ("ffmpeg-gpu" if self._ff.gpu else "ffmpeg") if self._ff else "opencv"

    def _try_ffmpeg(self, f0: np.ndarray, use_gpu: bool) -> Optional[_FFmpeg]:
        """ffmpeg reader, if it sees the same frames OpenCV does."""
        try:
            ff = _FFmpeg(self.path, use_gpu)
            first = next(ff.read(r"eq(n\,0)", extra_cw=self.rotate, limit=1), None)
        except Exception as e:
            log.debug("ffmpeg reader unavailable: %s", e)
            return None
        if first is None or first.shape != f0.shape:
            log.debug("ffmpeg frames %s don't match OpenCV's %s; using OpenCV",
                      None if first is None else first.shape, f0.shape)
            return None
        return ff

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

    def frames(self, step: int = 1, width: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_index, BGR frame) for every ``step``-th decodable frame,
        scaled to ``width`` pixels wide when given.

        Undecodable frames are skipped; a long run of failures is treated as
        the end of a truncated file rather than an error.
        """
        step = max(1, int(step))
        if self._ff is not None:
            k = -1
            for k, f in enumerate(self._ff.read(rf"not(mod(n\,{step}))", width, self.rotate)):
                yield k * step, f
            if k >= 0:
                return
            log.warning("ffmpeg returned no frames; falling back to OpenCV")
        for idx, f in self._frames_cv(step):
            if width and f.shape[1] != width:
                f = cv2.resize(f, (int(width), max(8, int(round(f.shape[0] * width / f.shape[1])))),
                               interpolation=cv2.INTER_AREA)
            yield idx, f

    def _frames_cv(self, step: int) -> Iterator[Tuple[int, np.ndarray]]:
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
        if self._ff is not None:
            got = 0
            for c in range(0, len(wanted), 400):  # keeps the select expression short
                chunk = wanted[c:c + 400]
                expr = "+".join(rf"eq(n\,{i})" for i in chunk)
                for i, f in zip(chunk, self._ff.read(expr, None, self.rotate, limit=len(chunk))):
                    got += 1
                    yield i, f
            if got:
                return
            log.warning("ffmpeg returned no frames; falling back to OpenCV")
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
