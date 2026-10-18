# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `piffler fix`: redo chosen pages of a finished book without reprocessing it, by hand
  (`--pages 68 --set gutter=0.53`, `--drop`, `--keep`) or automatically (`--auto`), which tries
  a few variants per flagged page and keeps the best-scoring one. Settings are saved in
  `overrides.json` and applied by every later run.
- Each page gets a stable id (`s142-left`) and a sidecar file with its words and the context
  it was made from.

## [0.1.0] - 2026-10-17

First public release.

### Added

- `piffler` command-line tool and Python API (`piffler.process_video`, `piffler.Config`)
  that turn a phone video of a book being flipped through into one image per page.
- Detection of steady page holds amid page-turn motion, with deduplication so each
  page is captured once.
- Finger removal using pixels from other frames, page cropping and perspective
  correction, spread splitting at the gutter, and dewarping of curved pages.
- Clean scans (flattened, white paper, dark text) alongside the source photos.
- Review flags for pages that need a re-shoot (e.g. `finger_may_cover_text`,
  `short_hold`, `low_ocr_confidence`).
- OCR via RapidOCR on ONNX Runtime (DirectML GPU acceleration on Windows), with
  per-page text output.
- Searchable PDF export.
