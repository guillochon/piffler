# Repairing pages with `piffler fix`

`piffler fix OUT ...` redoes chosen pages of a finished output folder without reprocessing the book. It decodes only the frames those pages need, redoes them with the context they were made in, and rebuilds the numbering, text, PDF and manifest, reusing every other page. One page takes seconds; `--auto` on ~20 flagged pages takes ~15 minutes.

Everything set is saved in `OUT/overrides.json` (keyed by hold, e.g. `s142`, or page id, e.g. `s142-left`) and applied by every later run, full or fix. To undo a setting, edit or delete that entry and run `piffler fix OUT --pages N` (or the next full run picks it up).

## Addressing pages

- By number: `--pages 68,88` (current numbering, which shifts when pages are dropped or restored).
- By id: `--pages s142-left` — hold 142 of the video, left half of the spread (`s142` alone for an unsplit page). Ids are stable; use them in scripts and when restoring dropped pages (dropped ids are listed under `dropped` in `manifest.json`).
- Settings apply to the whole hold: changing the split of a spread affects both its pages.

## Automatic: `--auto`

`piffler fix OUT --auto` (or `--auto --pages 68,88` for specific pages) matches each page's problem to variants, scores each (confidently read text, minus text a hand may hide, minus words cut at the page edge, minus text belonging to a neighbouring page) and keeps a variant only if it clearly beats the page as it is. It prints, per hold, the problem, the scores and what was applied.

Always look at the result afterwards (`review_book.py`): the score can prefer a page that reads more words but looks worse. If an auto change is wrong, remove that hold's entry from `overrides.json` and run `piffler fix OUT --pages N`.

## Settings (`--set KEY=VALUE`)

| Setting | Values | Fixes | Suggested |
|---|---|---|---|
| `gutter` | split position across the spread, 0-1 | facing-page text at the inner edge, words cut at the inner edge | read the current split from the page's `.json` sidecar (`ctx.split`), try ±0.02, then ±0.04. Typical values 0.47-0.53; it drifts toward ~0.52 later in a book as the stack of read pages grows on the left |
| `expand_page` | 0.01-0.06 | text or margins cut off at the outer edge | 0.03; 0.05 for badly cut pages |
| `outline` | `neighbors` | page outline wrong (book moved, a hand or page edge confused it) | use the book's usual position |
| `layout` | `single`, `spread` | covers or unsplit spreads | `spread` to force a split at the book's usual gutter; `single` for covers and inserts |
| `frames` | `all` | finger over text: look at every frame of the pause (up to ~30) instead of the best 5 | first thing to try for hands |
| `ref_frame` | a frame number | pick the frame yourself (see `--debug` pool images, which label each frame with its number and hand coverage) | when you can see a clear frame |
| `exclude_frames` | `[N, N]` | a bad frame keeps being used (blur, a hand) | — |
| `misfit` | 0.03-0.08 (default 0.08) | ghosted / doubled text after hand removal: stricter means fewer frames are trusted to fill in | 0.05, then 0.04 |
| `fill_fingers` | `false` | ghosting that `misfit` doesn't cure; leaves the hand visible instead | last resort |
| `dewarp` | `false` | wavy or distorted lines (picture pages, failed flattening) | — |
| `deshear` | `false` | column straightening made a page worse | — |
| `drop` | via `--drop N` | junk page, a repeat that wasn't caught | — |
| `keep` | via `--keep ID` | a real page dropped as a repeat (restores it, and it is never dropped again) | — |

Values are JSON: `--set frames=all --set misfit=0.05 --set exclude_frames=[10412,10416]`.

## Playbook by problem

1. **Hand over text** (`finger_may_cover_text`): `--set frames=all`; if still hidden, find a clean frame in `--debug` output and `--set ref_frame=N`. If the hand never moved, tell the user that page needs re-filming.
2. **Facing-page strip / words cut at the gutter**: `--set gutter=<ctx.split ± 0.02>`; check both pages of the spread afterwards.
3. **Cut at the outer edge**: `--set expand_page=0.03`, else `--set outline=neighbors`.
4. **Ghosting**: `--set misfit=0.05`, then `0.04`, then `--set fill_fingers=false`.
5. **Unsplit spread**: `--set layout=spread`.
6. **Junk or duplicate page**: `--drop N`. Check `book.txt` around it first, so a real page isn't lost.
7. **Missing page**: look at `dropped` in the manifest for its neighbours' time; `--keep ID` if it was dropped. If it was never captured (no pause on it), only re-filming helps; a full run with `--min-hold 0.25` may catch it.
