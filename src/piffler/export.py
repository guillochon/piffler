"""Write page images, text, a JSON manifest and an optional searchable PDF."""
from __future__ import annotations

import json
import logging
import os
from typing import List

import cv2
import numpy as np

log = logging.getLogger(__name__)


def write_pdf(path: str, pages, dpi: float = 150.0) -> bool:
    """One PDF page per book page, all the same size (the median page), with
    the scan centred on it and an invisible, searchable text layer."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfbase.pdfmetrics import stringWidth
    except ImportError:
        log.info("reportlab not installed; skipping PDF (pip install reportlab)")
        return False
    items = []
    for p in pages:
        img = cv2.imread(p.image_path, cv2.IMREAD_UNCHANGED) if p.image_path else None
        if img is None:
            continue
        sc = getattr(p, "image_scale", 1.0) or 1.0
        items.append((p, img.shape[1] / sc, img.shape[0] / sc))  # size in OCR pixels
    if not items:
        return False
    k = 72.0 / dpi  # points per OCR pixel
    pw = float(np.median([w for _, w, _ in items])) * k
    ph = float(np.median([h for _, _, h in items])) * k
    c = canvas.Canvas(path, pagesize=(pw, ph))
    c.setTitle(os.path.splitext(os.path.basename(path))[0])
    for p, w, h in items:
        f = min(pw / (w * k), ph / (h * k))  # fit, keeping the page's own shape
        iw, ih = w * k * f, h * k * f
        ox, oy = (pw - iw) / 2, (ph - ih) / 2
        c.drawImage(ImageReader(p.image_path), ox, oy, iw, ih)
        s = k * f  # points per OCR pixel on this page
        for wd in p.words:
            x0, x1 = wd.box[:, 0].min(), wd.box[:, 0].max()
            y0, y1 = wd.box[:, 1].min(), wd.box[:, 1].max()
            fs = max(1.0, (y1 - y0) * s * 0.8)
            txt = wd.text.encode("latin-1", "replace").decode("latin-1")
            tw = stringWidth(txt, "Helvetica", fs) or 1.0
            t = c.beginText()
            t.setTextRenderMode(3)  # invisible
            t.setFont("Helvetica", fs)
            t.setHorizScale(100.0 * (x1 - x0) * s / tw)
            t.setTextOrigin(ox + x0 * s, oy + ih - y1 * s + 0.2 * (y1 - y0) * s)
            t.textOut(txt)
            c.drawText(t)
        c.showPage()
    c.save()
    return True


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
