from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import Config


def _value(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def _split(v):
    return [x for part in (v or []) for x in part.split(",") if x.strip()]


def _progress(stage, frac):
    sys.stderr.write(f"\r{stage:>10}: {frac * 100:5.1f}%")
    if frac >= 1.0:
        sys.stderr.write("\n")
    sys.stderr.flush()


def fix_main(argv) -> int:
    ap = argparse.ArgumentParser(
        prog="piffler fix",
        description="Redo chosen pages of a finished book with different settings, without reprocessing it. "
                    "Settings are saved in OUT/overrides.json and applied by every later run.",
        epilog="settings: ref_frame=N, exclude_frames=[N,..], frames=all, fill_fingers=false, misfit=0.05, "
               "expand_page=0.03, outline=neighbors, layout=single|spread, gutter=0.52, dewarp=false, "
               "deshear=false")
    ap.add_argument("out", help="output folder of an earlier run")
    ap.add_argument("--pages", action="append", help="page numbers or ids (s142-left), comma-separated")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="setting for --pages")
    ap.add_argument("--drop", action="append", help="pages to remove")
    ap.add_argument("--keep", action="append", help="pages never to drop (ids of dropped pages restore them)")
    ap.add_argument("--auto", action="store_true",
                    help="try variants on flagged pages (or on --pages) and keep what scores best")
    ap.add_argument("--video", help="the video, if it has moved since the run")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    settings = {}
    for kv in a.set:
        if "=" not in kv:
            ap.error(f"--set needs KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        settings[k.strip()] = _value(v.strip())
    if settings and not a.pages:
        ap.error("--set needs --pages")
    from .fix import fix
    try:
        rep = fix(a.out, _split(a.pages), settings, _split(a.drop), _split(a.keep), a.auto, a.video,
                  progress=_progress)
    except (ValueError, RuntimeError, OSError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    for pid, st in rep["set"].items():
        print(f"{pid}: set {st}")
    for pid in rep["dropped"]:
        print(f"{pid}: dropped")
    for pid in rep["restored"]:
        print(f"{pid}: kept")
    for e in rep["auto"]:
        res = f"applied {e['applied']}" if e["applied"] else "kept as is"
        print(f"{e['hold']} ({', '.join(e['problems'])}): tried {e['tried']}, score {e['baseline']} -> "
              f"{e['best']}, {res}")
    if a.auto and not rep["auto"]:
        print(f"Nothing to try: {rep.get('checked', 0)} page(s) checked, none with a problem recovery can address.")
    if "pages" in rep:
        print(f"{rep['pages']} pages; worth a look: {rep['needs_review']}")
    return 0


def skill_dir() -> str:
    """Where the bundled agent skill lives inside the installed package."""
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "skill")


def skill_main(argv) -> int:
    import os
    import shutil
    ap = argparse.ArgumentParser(
        prog="piffler skill",
        description="The piffler agent skill: instructions and suggested settings for AI coding agents "
                    "(Claude Code and other tools that read SKILL.md skills).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ins = sub.add_parser("install", help="copy the skill where agents find it")
    where = ins.add_mutually_exclusive_group()
    where.add_argument("--project", action="store_true",
                       help="into ./.claude/skills/piffler (this project only) instead of your user folder")
    where.add_argument("--dest", help="into this folder instead (the skill goes in DEST/piffler)")
    ins.add_argument("--force", action="store_true", help="replace an existing copy")
    sub.add_parser("path", help="print where the bundled skill is")
    a = ap.parse_args(argv)
    src = skill_dir()
    if a.cmd == "path":
        print(src)
        return 0
    if a.dest:
        base = a.dest
    elif a.project:
        base = os.path.join(os.getcwd(), ".claude", "skills")
    else:
        base = os.path.join(os.path.expanduser("~"), ".claude", "skills")
    dst = os.path.join(base, "piffler")
    if os.path.exists(dst):
        if not a.force:
            print(f"error: {dst} already exists (use --force to replace it)", file=sys.stderr)
            return 2
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"installed the piffler skill in {dst}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "fix":
        return fix_main(argv[1:])
    if argv and argv[0] == "skill":
        return skill_main(argv[1:])
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
    ap.add_argument("--config", action="append", default=[], metavar="KEY=VALUE",
                    help="set any Config field, e.g. --config motion_k=0.35 --config candidates_per_page=8 "
                         "(see the piffler skill's tuning reference)")
    ap.add_argument("--debug", action="store_true", help="save intermediate images and stats")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = Config(layout=a.layout, rotate=a.rotate, min_hold_s=a.min_hold, ocr=not a.no_ocr,
                 dewarp=not a.no_dewarp, remove_fingers=not a.no_fingers, dedupe=not a.no_dedupe,
                 pdf=not a.no_pdf, save_debug=a.debug, max_pages=a.max_pages, gpu=not a.cpu, scan=not a.no_scan, scan_style=a.scan_style)
    for kv in a.config:
        if "=" not in kv:
            ap.error(f"--config needs KEY=VALUE, got {kv!r}")
        k, v = (x.strip() for x in kv.split("=", 1))
        if k not in Config.__dataclass_fields__ or k == "extra":
            ap.error(f"unknown config field {k!r}; fields: {', '.join(f for f in Config.__dataclass_fields__ if f != 'extra')}")
        cur = getattr(cfg, k)
        val = _value(v)
        if isinstance(cur, bool) and not isinstance(val, bool):
            ap.error(f"{k} takes true or false")
        if isinstance(cur, (int, float)) and not isinstance(cur, bool) and not isinstance(val, (int, float)):
            ap.error(f"{k} takes a number")
        setattr(cfg, k, type(cur)(val) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else val)
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
