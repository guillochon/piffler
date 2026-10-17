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
    dedupe_similarity: float = 0.06   # ORB inlier ratio above which two holds are the same page
                                      # (real footage: same page 0.08-0.7, different pages < 0.02)

    # --- page geometry ----------------------------------------------------
    layout: str = "auto"              # "auto" | "single" | "spread" (two pages per frame)
    detect_long_side: int = 2400      # page detection runs on frames downscaled to this
    work_long_side: int = 2400        # the cropped page region is downscaled to at most this
    max_frame_side: int = 4096        # decoded frames larger than this are downscaled on arrival
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
    scan: bool = True                 # clean, high-contrast page scans (used in the PDF)
    scan_upscale: float = 2.0         # scans are this much larger than the page photo
    scan_style: str = "natural"       # "natural" (the page's own tones) or "bw" (crisp black on white)
    gpu: bool = True                  # decode video and run OCR on the GPU when one is available
    max_pages: int = 0                # stop after this many pages (0 = whole book)

    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
