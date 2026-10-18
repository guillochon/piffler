---
name: piffler
description: Digitize a book or document from a phone video of someone flipping through it, using the piffler package (page scans, OCR text, searchable PDF). Use when the user wants to scan, digitize or OCR a book, booklet, magazine or stack of pages from a video (.mp4, .mov), tune piffler's settings for a particular video, review piffler's output, or fix bad pages (fingers over text, cut-off text, duplicated or missing pages, wrong page splits) in a finished piffler output folder.
---

# Digitizing a book with piffler

piffler turns a video of pages being turned into one clean image per page, OCR text and a searchable PDF. A full run of a 200-page book takes about 25 minutes on a GPU, so work in this order: **look at the video, do a short trial, tune, run the whole book, then repair individual pages** with `piffler fix` (seconds per page) instead of re-running everything.

Check it's installed: `piffler --help` (install with `pip install piffler`; ffmpeg on PATH makes decoding much faster).

## 1. Look at the video first

```bash
ffprobe -v error -show_entries format=duration:stream=width,height,r_frame_rate -of default=nw=1 VIDEO
```

Grab a couple of frames (`ffmpeg -ss 30 -i VIDEO -frames:v 1 frame30.png`) and look at them. Note:
- **Layout**: two-page spreads (an open book) or one page per frame (loose sheets, close-up).
- **Background**: darker than the paper is what piffler expects; a white table is a known weak spot.
- **Camera**: on a stand (steady) or hand-held (moving).
- **Hands**: where they hold the book, whether they rest on text.
- **Print**: small print or a book small in the frame needs more resolution.
- **Pace**: roughly how long each page is held still (the pause is how pages are found).

Pick starting settings from [reference/tuning.md](reference/tuning.md) ("Starting points by video").

## 2. Trial run on the first pages

```bash
piffler VIDEO -o OUT --max-pages 30 [flags from step 1]
python "$(piffler skill path)/scripts/review_book.py" OUT
```

(`piffler skill path` prints where the package keeps this skill; the review script is there too.) The analysis pass (finding the still moments) runs once and is cached in `OUT/analysis_cache.npz`, so later runs into the same folder skip it. `review_book.py` prints page-order problems (from the printed page numbers it reads), the review list, dropped pages and contact sheets; **open the contact sheets and look at them** — numbers alone miss cut-off text and ghosting. Match what you see to the symptom table in [reference/tuning.md](reference/tuning.md) and re-run the trial with adjusted settings. Change one or two settings at a time.

Settings that aren't flags are set with `--config KEY=VALUE` (repeatable), e.g. `--config motion_k=0.35`.

## 3. Full run

```bash
piffler VIDEO -o OUT [tuned flags]
python "$(piffler skill path)/scripts/review_book.py" OUT
```

Use a fresh folder or the trial folder (the cache is reused either way when settings that affect analysis haven't changed). Check the summary: page numbers should run in order, the review list should be short.

## 4. Repair pages, don't re-run

```bash
piffler fix OUT --auto                       # tries variants on every flagged page, keeps winners
python "$(piffler skill path)/scripts/review_book.py" OUT  # look again
piffler fix OUT --pages 68,88 --set gutter=0.52   # then by hand, per page
piffler fix OUT --drop 143                   # junk page
piffler fix OUT --keep s150-left             # restore a page that was dropped as a repeat
```

Pages are addressed by number (current numbering) or by stable id (`s142-left`: hold 142, left half), shown in `manifest.json`. Everything set is saved in `OUT/overrides.json` and re-applied by every later run, full or fix. See [reference/recovery.md](reference/recovery.md) for which setting fixes which problem.

## 5. Report to the user

Give the page count, whether page numbering is continuous, what was fixed, and the pages still worth a human look (with why). Don't claim pages are perfect from the scores alone; say what you looked at.

## Things to know

- Output: `OUT/book.pdf`, `OUT/book.txt`, `OUT/pages/page_NNNN.jpg` (clean scan), `..._photo.jpg` (the page photo), `.txt`, `.json` (words and context), `OUT/manifest.json` (per page: id, flags, info; `needs_review`; `dropped` with reasons).
- `needs_review` flags: `finger_may_cover_text` (a hand hid text no other frame showed), `page_from_neighbors` (page outline guessed from neighbouring pages), `possible_duplicate`, `low_ocr_confidence`, `gutter_guessed`. Informational only: `no_text_found` (blank page), `short_hold`, `gutter_from_neighbors`, `page_near_neighbors`, `adjusted` (has overrides).
- Pages whose text repeats a nearby page (the reader lifted a page mid-turn, or flipped back) are dropped automatically and listed under `dropped` in the manifest; `--keep ID` restores one.
- A blank page that comes out blank is correct; don't try to "fix" it.
- If the user can re-film, the cheapest fix of all is a pause of about half a second on each page, hands on the margins, phone on a stand.
