# Piffler the Page Riffler

Turn a phone video of you flipping through a book into clean page scans, plain text, and a searchable PDF. The book stays intact: no flatbed, no pressing the spine.

```
pip install "piffler[pdf]"
piffler my_book.mp4 -o my_book/
```

ffmpeg on your PATH is recommended (faster, GPU-accelerated decoding); without it, OpenCV decodes the video.

Output:

```
my_book/
  pages/page_0001.png        clean scan: flattened, finger-free, white paper, dark text
  pages/page_0001_photo.jpg  the same page as a colour photo, before clean-up
  pages/page_0001.txt        OCR text for that page
  book.txt                   all text, in page order
  book.pdf                   the clean scans with an invisible, searchable text layer
  manifest.json              per-page timing, quality metrics, and review flags
  analysis_cache.npz         frame analysis, reused when you re-run on the same video
```

Useful options: `--max-pages N` to try the first few pages, `--cpu` to stay off the GPU, `--no-scan` to put the photos in the PDF instead of clean scans.

## GPU

The GPU is used when there is one, with no setup:

- **Video decoding** runs through ffmpeg, on NVIDIA GPUs with NVDEC, which also scales frames down before analysis. Without a GPU it is still multi-threaded ffmpeg, and OpenCV is the fallback when ffmpeg isn't installed. On a 6¾-minute 4K iPhone video the analysis pass takes about 30 s, against about 15 min with OpenCV alone.
- **OCR** runs on onnxruntime's best provider: DirectML on Windows (any DirectX 12 GPU, installed by default), CUDA where `onnxruntime-gpu` is installed (`pip uninstall onnxruntime && pip install onnxruntime-gpu` on Linux), CoreML on macOS. About 4× faster than the CPU per page.

## How to film

Put the phone on a stand above the book with even light. Turn a page, then **pause about half a second** before turning the next one. That pause is how pages are found. Keep fingers on the margins if you can. If they move during the pause, they are erased using pixels from other frames. If you pause on the same page more than once, the frames of all those pauses are pooled, so a finger that covered text in one is filled from another. Two-page spreads and single pages both work (`--layout auto` decides). Slow-motion video is fine, since it is subsampled for analysis.

## What it does

1. **Finds still moments** (`segment.py`). The video is streamed once at about 15 fps, keeping only per-frame numbers (motion, sharpness, exposure, skin coverage), so memory stays flat for long videos. A page is "held" when motion drops below a threshold derived from that video's own motion distribution. This adapts to tripods, shaky hands, and compression noise without tuning. If no pauses are found, the threshold is relaxed, and as a last resort the sharpest frames are used.
2. **Picks the best frames.** Up to 5 candidates per hold are chosen, spread out in time, and scored by sharpness, exposure, and finger coverage. The page's reference frame is the one whose text block is least covered by fingers.
3. **Pools repeat holds.** If you pause on the same page more than once (a hesitation, or flipping back), ORB matching of the page itself recognises it, and the frames of every pause are pooled for that page.
4. **Removes fingers** (`hands.py`). Skin is detected with colour models plus two guards against aged paper, which is often skin-coloured to a naive detector. A pixel must differ from that frame's own paper colour and fall in skin's Lab hue range (about 20–62°, while yellowed paper sits around 65–90°). Washed-out skin under bright or cool light, which overlaps aged paper in colour, also counts when its blob reaches past the page, as a hand attached to an arm does. Independently of colour, the other frames of the page are aligned and their per-pixel median taken; whatever covers the page in this frame but not in most others (a hand of any skin tone, a sleeve, a pen) is removed too. Covered areas, plus the hand's cast shadow, are replaced with pixels from other aligned frames where the hand had moved, and a frame that doesn't line up with the reference around the hand is not used. Skin no frame can fill is painted with paper; the page it is on is flagged if that overlaps text.
5. **Crops and rectifies** (`page.py`). The page outline is detected and perspective-corrected, then spreads are split at the gutter (found from shading and the blank column between text blocks).
6. **Flattens curved pages** (`dewarp.py`). Text lines are detected and fitted with polynomials, then a smooth 2-D displacement field is solved by robust least squares and the page is remapped so the lines become straight. The field is clamped to where text was actually seen. If it doesn't measurably straighten the lines (picture pages, blank pages, blur), the page is left alone instead of being distorted.
7. **Evens out lighting.** Gutter shadows and lamp falloff are removed, and underexposed pages are lifted to near-white paper.
8. **Makes clean scans** (`scan.py`). Each page is cropped to the sheet of paper, its background divided out, upscaled 2× and sharpened, and set so paper is white and text strokes near-black. Large faint marks (finger traces, shadows) and marks touching the page edge (countertop, other pages' edges) are dropped; pictures are kept.
9. **Runs OCR** (`ocr.py`) with RapidOCR, on the colour page photo. It works with both the current `rapidocr` package and the legacy `rapidocr_onnxruntime`, rebuilds reading order (including two-column layouts), and auto-detects sideways or upside-down video once per book.

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

## Development

```
git clone https://github.com/guillochon/piffler && cd piffler
pip install -e ".[pdf,test]"
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
- **Little real footage so far.** Beyond the synthetic test video, it has been tuned on one real video (a French pocket book on a kitchen counter, 4K iPhone). Sizes are relative to the page and its text, and thresholds to the paper's brightness, so book size, distance and exposure should matter little, but expect some tuning on other books, lighting and backgrounds.
- **Background must be darker than the paper.** Pages are found as bright regions with print on them; on a white table, or with a bright object touching the book, the page outline can be wrong.
- **Phone on a stand works best.** Pages with little print borrow the book's position from neighbouring pages; with a hand-held phone this is switched off automatically, so blank and photo pages may be found less reliably.
- **Scan resolution is limited by the camera.** A pocket book filmed from above gets roughly 140 dpi per page from a 4K frame, so the clean scans are upscaled 2× and sharpened. Filming closer gives sharper scans.
