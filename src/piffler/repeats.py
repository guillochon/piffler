"""Drop pages that repeat another page: holds caught mid-turn or after a flip-back.

When the reader lifts a page to turn it, or goes back a page, the camera sees a
still book for a moment and that hold becomes a page. Its frames look like a
real page, so the evidence is in the content: its text is already on a
neighbouring page. The worse copy is dropped, and so is its spread partner if
that holds nothing new (the lifted page shows countertop or air there).
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Set, Tuple


def _bigrams(text: str) -> Set[Tuple[str, str]]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    w = re.findall(r"[^\W\d_]{3,}", t)
    return set(zip(w, w[1:]))


def _badness(p, spread_book: bool) -> tuple:
    """Higher is worse: which copy of a repeated page to drop."""
    unsplit = spread_book and p.info.get("half") is None
    guessed = bool({"page_from_neighbors", "short_hold"} & set(p.flags))
    return (unsplit, guessed, len(p.flags), -p.mean_score, -len(p.text))


def drop_repeats(pages: list, window: int = 4, min_bigrams: int = 20,
                 overlap: float = 0.4) -> Tuple[list, List[dict]]:
    """Pages in output order -> (kept pages, dropped [{page, reason, ...}])."""
    B = [_bigrams(p.text) for p in pages]
    spread_book = sum(p.info.get("half") is not None for p in pages) > 0.5 * max(1, len(pages))
    dropped = {}  # index -> reason
    later = set()  # dropped copies that came after the copy kept (a re-hold, not the original)
    changed = True
    while changed:
        changed = False
        alive = [i for i in range(len(pages)) if i not in dropped]
        for a, i in enumerate(alive):
            for j in alive[a + 1:a + 1 + window]:
                small = min(len(B[i]), len(B[j]))
                if small < min_bigrams:
                    continue
                shared = len(B[i] & B[j])
                if max(shared / len(B[i]), shared / len(B[j])) < overlap:
                    continue
                worse = max((i, j), key=lambda k: _badness(pages[k], spread_book))
                other = pages[j if worse == i else i]
                if pages[worse].frame_index > other.frame_index:
                    later.add(worse)
                dropped[worse] = f"repeats the page at frame {other.frame_index} ({other.info.get('half') or 'whole'})"
                changed = True
                break
            if changed:
                break
    # a dropped repeat's spread partner came from the same hold: drop it too
    # when it shows nothing of its own (little or no text)
    for k in [k for k in dropped if k in later]:
        for m, p in enumerate(pages):
            if m not in dropped and p.segment == pages[k].segment and len(B[m]) < min_bigrams:
                dropped[m] = "partner of a repeated page, little text of its own"
    # the book wasn't found in the frame at all and nothing on it is readable
    for m, p in enumerate(pages):
        if m not in dropped and "page_from_neighbors" in p.flags and len(p.text.strip()) < 20:
            dropped[m] = "book not found in this frame, nothing readable"
    kept = [p for m, p in enumerate(pages) if m not in dropped]
    info = [{"segment": pages[m].segment, "frame_index": pages[m].frame_index,
             "time_s": round(pages[m].time_s, 2), "half": pages[m].info.get("half"),
             "chars": len(pages[m].text), "reason": r} for m, r in sorted(dropped.items())]
    return kept, info
