from __future__ import annotations

import argparse
import logging
import sys

from .config import Config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="piffler",
                                 description="Turn a page-flipping video into page scans + OCR text.")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="piffler_out", help="output directory")
    ap.add_argument("--layout", choices=["auto", "single", "spread"], default="auto",
                    help="one page per frame, two-page spreads, or detect")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="force rotation if the video is sideways (default: auto-detect)")
    ap.add_argument("--min-hold", type=float, default=0.35, help="seconds a page must be still")
    ap.add_argument("--max-pages", type=int, default=0, help="stop after N pages (default: all)")
    ap.add_argument("--cpu", action="store_true", help="don't use the GPU even if one is available")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-dewarp", action="store_true")
    ap.add_argument("--no-fingers", action="store_true", help="don't attempt finger removal")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-scan", action="store_true", help="use the page photos instead of clean scans")
    ap.add_argument("--scan-style", choices=["natural", "bw"], default="natural",
                    help="natural: the page's own paper tone and soft text; bw: crisp black on white")
    ap.add_argument("--debug", action="store_true", help="save intermediate images and stats")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = Config(layout=a.layout, rotate=a.rotate, min_hold_s=a.min_hold, ocr=not a.no_ocr,
                 dewarp=not a.no_dewarp, remove_fingers=not a.no_fingers, dedupe=not a.no_dedupe,
                 pdf=not a.no_pdf, save_debug=a.debug, max_pages=a.max_pages, gpu=not a.cpu, scan=not a.no_scan, scan_style=a.scan_style)
    from .pipeline import process_video
    from .video import VideoError
    last = [None]

    def prog(stage, frac):
        if stage != last[0] or frac >= 1.0:
            last[0] = stage
        sys.stderr.write(f"\r{stage:>10}: {frac * 100:5.1f}%")
        sys.stderr.flush()
        if frac >= 1.0:
            sys.stderr.write("\n")

    try:
        pages = process_video(a.video, a.out, cfg, progress=prog)
    except VideoError as e:
        sys.stderr.write(f"\nerror: {e}\n")
        return 2
    review = [p.page_no for p in pages if p.needs_review]
    print(f"\n{len(pages)} pages written to {a.out}/")
    if review:
        print(f"Pages worth a look (see manifest.json for reasons): {review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
