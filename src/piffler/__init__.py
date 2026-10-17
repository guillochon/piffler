"""Piffler the Page Riffler: video of a book being flipped -> page images + OCR text."""
from .config import Config

__version__ = "0.1.0"


def __getattr__(name):  # lazy: avoid importing heavy OCR deps on `import piffler`
    if name in ("process_video", "PageResult"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)


__all__ = ["Config", "process_video", "PageResult"]
