# Tuning piffler for a video

Defaults were tuned on a 4K iPhone video of a ~200-page pocket book (two-page spreads) on a speckled kitchen counter, phone on a stand, page held ~1-3 s, hands on the margins. Start from the defaults unless the video differs from that in a way listed below.

Flags are given as CLI options; everything else is `--config KEY=VALUE`.

## Settings

| Setting | Default | What it does | Raise it when | Lower it when |
|---|---|---|---|---|
| `--min-hold` (`min_hold_s`) | 0.35 s | shortest pause that counts as a page | junk pages from half-turned pages, very slow page turns: 0.5-0.6 | the reader barely pauses and pages are missing: 0.2-0.25 |
| `motion_k` | 0.25 | how still "still" is, relative to the video's own motion (0 = strictest) | hand-held or shaky video, compression noise, pages missed: 0.35-0.45 | tripod and a calm reader, but turns are merged into holds: 0.15-0.2 |
| `analysis_fps` | 15 | frames per second examined to find pauses | very fast page turns: 30 | very long videos where analysis is slow: 10 |
| `candidates_per_page` | 5 | frames kept per pause, to pick the least-covered one and fill in hands | fingers often rest on text: 8-10 (slower) | no hands in shot: 3 |
| `--layout` | auto | spread / single page detection | force `spread` for an open book if pages come out unsplit; `single` for loose sheets, phone close-ups, or covers | — |
| `--rotate` | auto | extra rotation | text comes out sideways or upside down: 90/180/270 | — |
| `dedupe_similarity` | 0.06 | how alike two pauses must look to be the same page (same page scores 0.08-0.7, different pages below 0.02) | distinct pages get merged (many blank or near-identical pages, e.g. a photo album): 0.1-0.15 | repeat pauses on one page aren't pooled: 0.04 |
| `--no-dedupe` | off | turns off both pooling of repeat pauses and dropping of repeated pages | the book genuinely repeats pages or text (refrains, forms) and pages vanish | — |
| `skin_paper_delta` | 14 | how different from the paper a pixel must be to count as skin | yellowed or tinted paper is erased as if it were a hand: 18-22 | pale hands or bright light, fingers left in: 10-12 |
| `--no-fingers` | off | skip hand removal | no hands in the shot (faster, and nothing can be wrongly erased) | — |
| `work_long_side` | 2400 px | resolution the page is processed at | small print, or the book small in the frame (pages under ~1000 px tall): 3200 | very large pages, slow runs: 1800 |
| `detect_long_side` | 2400 px | resolution the page is found at | rarely; tiny book in a huge frame: 3200 | — |
| `lang_det_long_side` | 1600 px | resolution OCR reads at | small print with OCR errors: 2400 (slower OCR) | — |
| `ocr_min_score` | 0.5 | OCR words below this confidence are discarded | lots of garbage words: 0.6-0.7 | faint print losing real words: 0.4 |
| `--no-dewarp` | off | skip line straightening and column de-leaning | flat sheets on a table (nothing to flatten) | — |
| `--scan-style` | natural | look of the clean scans: the paper's own tint, or crisp black on white | text-only archival copy, smaller files: `bw` | — |
| `scan_upscale` | 2.0 | scan size relative to the page photo | small print: 2.5 | large-format pages, or a smaller PDF: 1.5 |
| `jpeg_quality` | 92 | page photo quality | — | smaller output: 85 |
| `--max-pages` | all | stop after N pages | trial runs: 20-40 | — |
| `--cpu` | off | don't use the GPU | GPU errors | — |
| `--debug` | off | saves intermediate images to `OUT/debug/` (per-pause frame pools with hands in red, before/after finger removal, raw pages) | diagnosing a specific problem | — |

Changing `analysis_fps`, `analysis_width` or `--rotate` re-runs the analysis pass; the others reuse the cache.

## Starting points by video

| Video | Start with |
|---|---|
| Open book, phone on a stand, dark background (the default case) | defaults |
| Hand-held phone | `--config motion_k=0.4 --min-hold 0.3`; expect more pages to need `piffler fix` |
| Fast page turner (under ~0.5 s per pause) | `--min-hold 0.25 --config analysis_fps=30` |
| Slow, deliberate reader, junk half-turned pages | `--min-hold 0.5` |
| Fingers resting on text a lot | `--config candidates_per_page=9`, then `piffler fix --auto` |
| Loose sheets / one page per frame | `--layout single` |
| Flat documents (letters, receipts) on a table, no hands | `--layout single --no-dewarp --no-fingers` |
| Old, yellowed or tinted paper | `--config skin_paper_delta=20` |
| Small print, or book small in the frame | `--config work_long_side=3200 --config lang_det_long_side=2400 --config scan_upscale=2.5` |
| Photo-heavy book (albums, catalogues) | `--config dedupe_similarity=0.1` if pages merge; keep `--scan-style natural` |
| Text-only book for archiving | `--scan-style bw` |
| Very long video, quick look first | `--max-pages 30` |

## Symptoms → what to change

| What you see | Likely cause | Try |
|---|---|---|
| Pages missing (gap in printed page numbers) | pause too short or motion threshold too strict | `--min-hold 0.25`, `--config motion_k=0.35`; check the gap's video time in the manifest neighbours |
| The same page twice, a few pages apart | the reader flipped back; not caught | `piffler fix --drop N` (or lower `dedupe_similarity` for the whole run) |
| Two pages merged into one, or a page lost | dedupe too loose | `--config dedupe_similarity=0.1` |
| Speckled / countertop images as pages | half-turned pauses | usually dropped automatically; otherwise `--min-hold 0.5` or `piffler fix --drop` |
| Whole spread on one page (not split) | aspect/gutter unclear | `--layout spread`, or `piffler fix --pages N --set layout=spread` |
| Text from the facing page at the inner edge | split slightly off | `piffler fix --pages N --set gutter=...` (see recovery.md) |
| Text cut off at an outer edge | page outline too tight | `piffler fix --pages N --set expand_page=0.03` |
| Finger or hand over text | hand never moved during that pause | `piffler fix --pages N --set frames=all`; for the whole run `candidates_per_page=9`; otherwise re-film that page |
| Pale ghost of a hand in the margin | hand detected only partly | lower `skin_paper_delta` to 11 |
| Paper erased in patches, as if a hand were there | paper colour taken for skin | raise `skin_paper_delta` to 18-22 |
| Doubled / ghosted text | a frame that didn't line up filled in a hand | `piffler fix --pages N --set misfit=0.04` (or `fill_fingers=false`) |
| Text column leaning | page rectified askew | usually corrected; `piffler fix --pages N --set deshear=false` if it made it worse |
| Wavy or distorted lines | dewarp misfit (picture pages, very curved pages) | `piffler fix --pages N --set dewarp=false` |
| Text sideways / upside down | rotation | `--rotate 90` (or 180/270) |
| Lots of OCR garbage on clean pages | small print | `--config lang_det_long_side=2400 --config work_long_side=3200` |
| Run is slow | big frames, CPU OCR | check the log says `ffmpeg-gpu` and `OCR device: dml/cuda`; `--config work_long_side=1800` |
