"""Summarize a piffler output folder and make contact sheets to look at.

    python review_book.py OUT [--sheet-size 40] [--no-sheets]

Prints: page count; whether the printed page numbers (read by OCR) run in
order, and where they don't; the pages flagged for review and why; pages
dropped as repeats; and writes contact sheets to OUT/review/:
  review_flagged.jpg     the flagged pages, large enough to judge
  sheet_001.jpg, ...     every page, small, numbered (a red "U" marks an unsplit spread)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import cv2
import numpy as np

REVIEW = {"finger_may_cover_text", "low_quality_frame", "page_not_found", "page_from_neighbors",
          "processing_error", "gutter_guessed", "no_still_segments", "frame_decode_failed",
          "possible_duplicate", "low_ocr_confidence", "dewarp_failed"}


def printed_number(text: str):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for l in lines[-1:] + lines[:1]:  # page numbers sit on the last or first line
        if re.fullmatch(r"\d{1,4}", l):
            return int(l)
    return None


def tile(path: str, h: int, label: str, mark: bool = False):
    im = cv2.imread(path) if path else None
    if im is None:
        im = np.full((h, int(h * 0.65), 3), 200, np.uint8)
    im = cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h), interpolation=cv2.INTER_AREA)
    cv2.putText(im, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    if mark:
        cv2.putText(im, "U", (im.shape[1] - 18, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return cv2.copyMakeBorder(im, 0, 0, 0, 3, cv2.BORDER_CONSTANT, value=(90, 90, 90))


def grid(tiles, cols):
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0] + 3, 0, w - t.shape[1], cv2.BORDER_CONSTANT,
                                value=(90, 90, 90)) for t in tiles]
    tiles += [np.full_like(tiles[0], 90)] * (-len(tiles) % cols)
    return np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--sheet-size", type=int, default=40)
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args(argv)
    with open(os.path.join(a.out, "manifest.json"), encoding="utf-8") as f:
        m = json.load(f)
    P = m["pages"]
    texts = {}
    for p in P:
        try:
            with open(os.path.join(a.out, "pages", f"page_{p['page']:04d}.txt"), encoding="utf-8") as f:
                texts[p["page"]] = f.read()
        except OSError:
            texts[p["page"]] = ""
    last_t = P[-1]["time_s"] if P else 0
    print(f"{len(P)} pages (video up to {last_t:.0f} s), {len(m.get('dropped', []))} dropped as repeats/junk, "
          f"run took {m.get('seconds', 0):.0f} s")

    # page order from printed numbers
    nums = [(p["page"], printed_number(texts[p["page"]])) for p in P]
    # years in headings and the like can't be page numbers of this book
    known = [(n, v) for n, v in nums if v is not None and v <= 2 * len(P) + 20]
    if known:
        offs = collections.Counter(v - n for n, v in known)
        off, cnt = offs.most_common(1)[0]
        print(f"printed page numbers read on {len(known)}/{len(P)} pages; {cnt} fit 'printed = page {off:+d}'")
        # where the offset changes and stays changed (pages gained or lost there);
        # a single odd reading between agreeing neighbours is an OCR misread
        runs, cur = [], off
        for k, (n, v) in enumerate(known):
            o = v - n
            nxt = known[k + 1][1] - known[k + 1][0] if k + 1 < len(known) else None
            if o != cur and o == nxt:
                runs.append((n, v, o))
                cur = o
        prev_o = off
        for n, v, o in runs[:25]:
            d = o - prev_o
            hint = (f"{d} page(s) missing" if d > 0 else f"{-d} extra page(s)") + " shortly before here"
            print(f"  from page {n} (printed {v}) the offset is {o:+d}: {hint}")
            prev_o = o
        if len(runs) > 25:
            print(f"  ... {len(runs) - 25} more")
        seen = collections.Counter(v for _, v in known)
        rep = sorted(v for v, c in seen.items() if c > 1)
        if rep:
            print(f"  printed numbers seen twice (repeats or misreads): {rep[:30]}")
    else:
        print("no printed page numbers read (unnumbered document?)")

    # review list
    flagged = m.get("needs_review", [])
    print(f"worth a look: {len(flagged)} pages")
    by_no = {p["page"]: p for p in P}
    for n in flagged:
        p = by_no[n]
        why = sorted(set(p["flags"]) & REVIEW)
        extra = ""
        if p["info"].get("hidden_text_frac"):
            extra = f", hidden text {100 * p['info']['hidden_text_frac']:.2f}%"
        print(f"  page {n:4d} {p['id']:<12} {', '.join(why)} ({p['chars']} chars{extra})")
    for d in m.get("dropped", [])[:40]:
        print(f"  dropped {d.get('id', '?'):<12} t={d.get('time_s', 0):6.1f}s  {d['reason']}")
    adjusted = [p["page"] for p in P if "adjusted" in p["flags"]]
    if adjusted:
        print(f"pages with overrides: {adjusted}")
    unsplit = [p["page"] for p in P if p["info"].get("half") is None]
    spreads = sum(p["info"].get("half") is not None for p in P)
    if spreads > len(P) / 2 and unsplit:
        print(f"unsplit pages in a book of spreads (covers, or a missed split): {unsplit}")

    if a.no_sheets or not P:
        return 0
    rd = os.path.join(a.out, "review")
    os.makedirs(rd, exist_ok=True)
    img = lambda p: os.path.join(a.out, p.get("image") or p.get("photo") or "")
    if flagged:
        g = grid([tile(img(by_no[n]), 420, f"{n} {by_no[n]['id']}") for n in flagged], 8)
        cv2.imwrite(os.path.join(rd, "review_flagged.jpg"), g)
    spread_book = spreads > len(P) / 2
    for s in range(0, len(P), a.sheet_size):
        chunk = P[s:s + a.sheet_size]
        g = grid([tile(img(p), 260, str(p["page"]), spread_book and p["info"].get("half") is None)
                  for p in chunk], 10)
        cv2.imwrite(os.path.join(rd, f"sheet_{s + 1:03d}.jpg"), g)
    print(f"contact sheets in {rd}/ (review_flagged.jpg, sheet_*.jpg): open them and look")
    return 0


if __name__ == "__main__":
    sys.exit(main())
