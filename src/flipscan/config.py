from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # --- frame analysis -------------------------------------------------
    analysis_fps: float = 15.0        # subsample high-fps (e.g. 240fps slow-mo) video for analysis
    analysis_width: int = 320         # width of the downscaled frames used for motion analysis
    min_hold_s: float = 0.35          # a page must be still at least this long to be captured
    motion_k: float = 0.25            # stillness threshold = q20 + k * (q90 - q20) of motion signal
    candidates_per_page: int = 5      # frames kept per still segment (for finger removal / best pick)
    max_bad_reads: int = 30           # consecutive undecodable frames before we assume EOF/truncation
    rotate: int = 0                   # extra clockwise rotation (0/90/180/270) if metadata is wrong

    # --- duplicate handling ----------------------------------------------
    dedupe: bool = True
    dedupe_similarity: float = 0.12   # ORB inlier ratio above which two holds are the same page

    # --- page geometry ----------------------------------------------------
    layout: str = "auto"              # "auto" | "single" | "spread" (two pages per frame)
    work_long_side: int = 2400        # full-res frames are downscaled to this before processing
    dewarp: bool = True

    # --- fingers ---------------------------------------------------------
    remove_fingers: bool = True
    skin_paper_delta: float = 14.0    # min Lab distance from paper colour for a pixel to count as skin

    # --- OCR -------------------------------------------------------------
    ocr: bool = True
    ocr_min_score: float = 0.5
    lang_det_long_side: int = 1600    # downscale for OCR to keep it fast; text stays legible

    # --- output ------------------------------------------------------------
    save_debug: bool = False
    pdf: bool = True
    jpeg_quality: int = 92

    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
