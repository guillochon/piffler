"""RapidOCR wrapper: version-tolerant API, reading order, orientation detection."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Word:
    text: str
    score: float
    box: np.ndarray  # (4, 2) in image pixel coords

    @property
    def cy(self):
        return float(self.box[:, 1].mean())

    @property
    def x0(self):
        return float(self.box[:, 0].min())

    @property
    def height(self):
        return float(self.box[:, 1].max() - self.box[:, 1].min())


class OCREngine:
    def __init__(self, long_side: int = 1600, min_score: float = 0.5):
        self.long_side = long_side
        self.min_score = min_score
        self._eng = None
        self._kind = None

    def _load(self):
        if self._eng is not None:
            return
        try:
            from rapidocr import RapidOCR  # rapidocr >= 2
            self._eng, self._kind = RapidOCR(), "v2"
        except ImportError:
            try:
                from rapidocr_onnxruntime import RapidOCR  # legacy package
                self._eng, self._kind = RapidOCR(), "v1"
            except ImportError as e:
                raise ImportError("RapidOCR is not installed: pip install rapidocr onnxruntime") from e
        for name in ("RapidOCR", "rapidocr"):  # quieten per-call INFO spam
            logging.getLogger(name).setLevel(logging.WARNING)

    def __call__(self, img: np.ndarray) -> List[Word]:
        self._load()
        h, w = img.shape[:2]
        s = min(1.0, self.long_side / float(max(h, w)))
        im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
        words: List[Word] = []
        try:
            if self._kind == "v2":
                r = self._eng(im)
                boxes, txts, scores = getattr(r, "boxes", None), getattr(r, "txts", None), getattr(r, "scores", None)
                if boxes is None or txts is None:
                    return []
                items = zip(boxes, txts, scores if scores is not None else [1.0] * len(txts))
            else:
                res, _ = self._eng(im)
                items = [(b, t, sc) for b, t, sc in (res or [])]
            for b, t, sc in items:
                if not t or not str(t).strip():
                    continue
                sc = float(sc)
                if sc < self.min_score:
                    continue
                words.append(Word(str(t).strip(), sc, np.asarray(b, np.float32).reshape(4, 2) / s))
        except Exception as e:
            log.warning("OCR failed on page: %s", e)
        return words


def reading_order(words: List[Word]) -> List[List[Word]]:
    """Group boxes into lines (top-to-bottom, left-to-right), handling 2 columns."""
    if not words:
        return []
    # column detection: a vertical gap free of boxes in the middle third
    xs0 = np.array([w.box[:, 0].min() for w in words]); xs1 = np.array([w.box[:, 0].max() for w in words])
    W = xs1.max()
    cols = [words]
    if W > 0 and len(words) > 12:
        grid = np.zeros(int(W) + 2, np.int32)
        for a, b in zip(xs0, xs1):
            grid[int(a): int(b) + 1] += 1
        lo, hi = int(W * 0.3), int(W * 0.7)
        mid = grid[lo:hi]
        if len(mid) and mid.min() == 0:
            zero = np.where(mid == 0)[0]
            gap = lo + int(np.median(zero))
            left = [w for w in words if w.box[:, 0].max() <= gap]
            right = [w for w in words if w.box[:, 0].min() >= gap]
            if len(left) >= 4 and len(right) >= 4 and len(left) + len(right) == len(words):
                cols = [left, right]
    lines: List[List[Word]] = []
    for col in cols:
        col = sorted(col, key=lambda w: w.cy)
        med_h = np.median([w.height for w in col]) or 10
        cur: List[Word] = []
        for w in col:
            if cur and abs(w.cy - np.mean([c.cy for c in cur])) > 0.5 * med_h:
                lines.append(sorted(cur, key=lambda c: c.x0)); cur = []
            cur.append(w)
        if cur:
            lines.append(sorted(cur, key=lambda c: c.x0))
    return lines


def words_to_text(words: List[Word]) -> str:
    return "\n".join(" ".join(w.text for w in ln) for ln in reading_order(words))


def ocr_quality(words: List[Word]) -> float:
    """Characters weighted by confidence; used to compare orientations."""
    return float(sum(len(w.text) * w.score ** 2 for w in words))


def best_rotation(img: np.ndarray, engine: OCREngine) -> int:
    """Return clockwise rotation (0/90/180/270) that makes the text upright."""
    h, w = img.shape[:2]
    s = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rots = {0: small, 90: cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(small, cv2.ROTATE_180), 270: cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE)}
    scores = {}
    for r, im in rots.items():
        scores[r] = ocr_quality(engine(im))
    best = max(scores, key=scores.get)
    # only rotate on clear evidence
    if best != 0 and scores[best] < 1.5 * scores[0] + 5:
        best = 0
    log.debug("orientation scores %s -> %d", scores, best)
    return best


def rotate(img: np.ndarray, deg: int) -> np.ndarray:
    m = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(img, m[deg]) if deg in m else img
