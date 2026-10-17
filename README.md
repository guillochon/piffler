# Piffler the Page Riffler

Turn a phone video of you flipping through a book into clean page scans, plain text, and a searchable PDF. The book stays intact: no flatbed, no pressing the spine.

```
pip install -e ".[pdf]"
piffler my_book.mp4 -o my_book/
```

Output:

```
my_book/
  pages/page_0001.jpg   cropped, flattened, finger-free page image
  pages/page_0001.txt   OCR text for that page
  book.txt              all text, in page order
  book.pdf              page images with an invisible, searchable text layer
  manifest.json         per-page timing, quality metrics, and review flags
```

## How to film

Put the phone on a stand above the book with even light. Turn a page, then **pause about half a second** before turning the next one. That pause is how pages are found. Keep fingers on the margins if you can. If they move during the pause, they are erased using pixels from other frames. Two-page spreads and single pages both work (`--layout auto` decides). Slow-motion video is fine, since it is subsampled for analysis.

## What it does

1. **Finds still moments** (`segment.py`). The video is streamed once at about 15 fps, keeping only per-frame numbers (motion, sharpness, exposure, skin coverage), so memory stays flat for long videos. A page is "held" when motion drops below a threshold derived from that video's own motion distribution. This adapts to tripods, shaky hands, and compression noise without tuning. If no pauses are found, the threshold is relaxed, and as a last resort the sharpest frames are used.
2. **Picks the best frames.** Up to 5 candidates per hold are chosen, spread out in time, and scored by sharpness, exposure, and finger coverage.
3. **Removes duplicates.** If you hesitate mid-page or flip back, ORB feature matching recognises the same page and keeps the better capture.
4. **Removes fingers** (`hands.py`). Skin is detected with colour models plus two guards against aged paper, which is often skin-coloured to a naive detector. A pixel must differ from that frame's own paper colour and fall in skin's Lab hue range (about 20–62°, while yellowed paper sits around 65–90°). Finger blobs are then replaced with pixels from other aligned frames where the finger had moved. Anything left is filled with paper texture, and the page is flagged if that area overlaps text.
5. **Crops and rectifies** (`page.py`). The page outline is detected and perspective-corrected, then spreads are split at the gutter (found from shading and the blank column between text blocks).
6. **Flattens curved pages** (`dewarp.py`). Text lines are detected and fitted with polynomials, then a smooth 2-D displacement field is solved by robust least squares and the page is remapped so the lines become straight. The field is clamped to where text was actually seen. If it doesn't measurably straighten the lines (picture pages, blank pages, blur), the page is left alone instead of being distorted.
7. **Evens out lighting.** Gutter shadows and lamp falloff are removed, and underexposed pages are lifted to near-white paper.
8. **Runs OCR** (`ocr.py`) with RapidOCR. It works with both the current `rapidocr` package and the legacy `rapidocr_onnxruntime`, rebuilds reading order (including two-column layouts), and auto-detects sideways or upside-down video once per book.

## Robustness

A bad page never stops the book. Each page is processed in isolation, and failures are recorded in `manifest.json` (`needs_review`, with reasons) instead of raised. Specific cases handled:

- **Truncated or corrupt files.** Undecodable frames are skipped, a long run of failures is treated as end-of-file, and if OpenCV can't decode the file at all it is transcoded with ffmpeg when available.
- **Bad metadata.** Missing or absurd fps and wrong frame counts are tolerated, and rotation metadata is honoured.
- **Exact frame fetching.** Frames are fetched by sequential decode, since seeking is unreliable on variable-frame-rate phone video.
- **Clear errors.** Empty, missing, or non-video files raise `VideoError` with a clear message.

Review flags include `finger_may_cover_text`, `short_hold`, `low_quality_frame`, `low_ocr_confidence`, `no_text_found`, `possible_duplicate`, `gutter_uncertain`, and `page_not_found`.

## Python API

```python
from piffler import Config, process_video

pages = process_video("book.mp4", "out/", Config(layout="spread", min_hold_s=0.5))
for p in pages:
    print(p.page_no, p.mean_score, p.flags, p.text[:80])
```

## Tests

```
pytest -m "not slow"   # 13 robustness tests, ~2 min
pytest -m slow         # end-to-end on a generated flip-through video, ~3 min
python tests/make_test_video.py demo.mp4   # make the test video yourself
```

The test video mixes real photos of curved, yellowed book pages (from the MIT-licensed [page-dewarp](https://github.com/lmmx/page-dewarp) examples) with rendered pages of known text. It simulates handheld shake, exposure drift, page-turn motion, fingers, a too-short hold, a mid-page hesitation, a dark page, a two-page spread, and a corrupted file tail. On it, every page is found once, and OCR matches ground truth at 100% character similarity on all unobstructed pages.

## Known limitations

- **Pauses are required.** Pages turned without any pause can be missed (they are reported, not silently dropped).
- **Dewarping is vertical only.** It straightens lines but does not undo horizontal compression right at the spine, so very tight bindings still look slightly squeezed near the gutter.
- **Covered text can't be recovered.** A finger that never moves off text leaves a gap, and the page is flagged so you can re-shoot it.
- **Untested on real footage.** All testing here used a synthetic flip-through video, so expect some tuning on real-world footage.
