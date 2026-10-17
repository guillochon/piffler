"""Write page images, text, a JSON manifest and an optional searchable PDF."""
from __future__ import annotations

import json
import logging
import os
from typing import List

import cv2
import numpy as np

log = logging.getLogger(__name__)


def write_pdf(path: str, pages) -> bool:
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfbase.pdfmetrics import stringWidth
    except ImportError:
        log.info("reportlab not installed; skipping PDF (pip install reportlab)")
        return False
    c = None
    for p in pages:
        img = cv2.imread(p.image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        dpi = 200.0
        pw, ph = w * 72.0 / dpi, h * 72.0 / dpi
        k = 72.0 / dpi
        if c is None:
            c = canvas.Canvas(path, pagesize=(pw, ph))
        else:
            c.setPageSize((pw, ph))
        c.drawImage(ImageReader(p.image_path), 0, 0, pw, ph)
        # invisible text layer so the PDF is searchable/selectable
        for wd in p.words:
            x0, x1 = wd.box[:, 0].min(), wd.box[:, 0].max()
            y0, y1 = wd.box[:, 1].min(), wd.box[:, 1].max()
            fs = max(1.0, (y1 - y0) * k * 0.8)
            txt = wd.text.encode("latin-1", "replace").decode("latin-1")
            tw = stringWidth(txt, "Helvetica", fs) or 1.0
            t = c.beginText()
            t.setTextRenderMode(3)
            t.setFont("Helvetica", fs)
            t.setHorizScale(100.0 * (x1 - x0) * k / tw)
            t.setTextOrigin(x0 * k, ph - y1 * k + 0.2 * (y1 - y0) * k)
            t.textOut(txt)
            c.drawText(t)
        c.showPage()
    if c is not None:
        c.save()
        return True
    return False


def write_outputs(out_dir: str, pages, meta: dict, make_pdf: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    book_txt = os.path.join(out_dir, "book.txt")
    with open(book_txt, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"===== Page {p.page_no} =====\n{p.text}\n\n")
    manifest = dict(meta)
    manifest["pages"] = [p.to_dict(out_dir) for p in pages]
    manifest["needs_review"] = [p.page_no for p in pages if p.needs_review]
    pdf_path = os.path.join(out_dir, "book.pdf")
    if make_pdf and pages:
        try:
            if write_pdf(pdf_path, pages):
                manifest["pdf"] = "book.pdf"
        except Exception as e:  # never lose the rest of the output over the PDF
            log.warning("PDF export failed: %s", e)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest
