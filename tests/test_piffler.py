"""Tests focus on messy real-world input: bad files, odd orientations,
no pauses, yellowed paper, plus an end-to-end accuracy check."""
from __future__ import annotations

import difflib
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import make_test_video as mtv  # noqa: E402

from piffler import Config  # noqa: E402
from piffler.dewarp import dewarp, find_lines, text_mask  # noqa: E402
from piffler.hands import skin_mask  # noqa: E402
from piffler.ocr import Word, reading_order, words_to_text  # noqa: E402
from piffler.page import detect_page, split_pages  # noqa: E402
from piffler.scan import clean_scan  # noqa: E402
from piffler.pipeline import process_video  # noqa: E402
from piffler.video import VideoError, VideoSource  # noqa: E402

DATA = os.path.join(HERE, "data")


def _sim(a, b):
    return difflib.SequenceMatcher(None, " ".join(a.split()), " ".join(b.split())).ratio()


def _write(path, frames, fps=30, faststart=True):
    raw = path + ".avi"
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", raw, "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p"]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    subprocess.run(cmd + [path], check=True)
    os.remove(raw)
    return path


def _page_frames(n_pages=2, hold=1.2, flip=0.4, fps=30, rot=None):
    cam = mtv.Camera()
    frames, truths = [], []
    for i in range(n_pages):
        img, truth = mtv.render_page(*mtv.TEXTS[i], 300 + i)
        base = mtv.place(mtv.curl(img, "left"), [[110, 330], [970, 345], [985, 1600], [95, 1590]])
        truths.append(truth)
        if i:
            for _ in range(int(flip * fps)):
                cam.step(5.0)
                frames.append(cam.apply(cv2.blur(base, (31, 1))))
        for _ in range(int(hold * fps)):
            cam.step(0.4)
            frames.append(cam.apply(base))
    frames = [cv2.resize(f, (540, 960)) for f in frames]
    if rot is not None:
        frames = [cv2.rotate(f, rot) for f in frames]
    return frames, truths


# ------------------------------------------------- real-kitchen-style scenes

def _cluttered_scene():
    """A small book page on speckled stone next to a big, bright, text-free pan."""
    rng = np.random.default_rng(3)
    h, w = 1920, 1080
    scene = np.clip(rng.normal(140, 35, (h, w, 1)), 0, 255).astype(np.uint8).repeat(3, 2)
    scene = cv2.GaussianBlur(scene, (3, 3), 0)
    cv2.circle(scene, (330, 420), 300, (238, 240, 236), -1)  # white enamel pan, brighter than the page
    page, _ = mtv.render_page(*mtv.TEXTS[1], 7)
    page = cv2.resize(page, (420, 600))
    scene[1150:1750, 450:870] = page
    return scene, (450, 1150, 870, 1750)


def test_page_found_by_text_not_by_brightness():
    scene, (x0, y0, x1, y1) = _cluttered_scene()
    quad, _, flags = detect_page(scene)
    assert "page_by_brightness" not in flags
    cx, cy = quad.mean(0)
    assert x0 < cx < x1 and y0 < cy < y1


def test_blank_spread_split_at_learned_gutter():
    spread = np.full((700, 800, 3), 235, np.uint8)  # two blank pages: no gutter to see
    halves, flags, sure = split_pages(spread, "auto", hint=0.45)
    assert len(halves) == 2 and sure is None
    assert abs(halves[0].shape[1] - 0.45 * 800) < 0.03 * 800


def test_max_pages_stops_early(tmp_path):
    frames, _ = _page_frames(3)
    p = _write(str(tmp_path / "v.mp4"), frames)
    pages = process_video(p, str(tmp_path / "out"), Config(ocr=False, pdf=False, max_pages=1))
    assert len(pages) == 1


def test_analysis_is_reused(tmp_path, monkeypatch):
    frames, _ = _page_frames(2)
    p = _write(str(tmp_path / "v.mp4"), frames)
    out = str(tmp_path / "out")
    first = process_video(p, out, Config(ocr=False, pdf=False))
    import piffler.pipeline as pl

    def boom(*a, **k):
        raise AssertionError("analysis should come from the cache")
    monkeypatch.setattr(pl, "analyze", boom)
    again = process_video(p, out, Config(ocr=False, pdf=False))
    assert len(again) == len(first)


def test_clean_scan_makes_paper_white_and_text_dark():
    page, _ = mtv.render_page(*mtv.TEXTS[0], 11)
    page = cv2.resize(page, (500, 720))
    dim = (page.astype(np.float32) * 0.8 + 25).astype(np.uint8)  # grey, low-contrast photo
    rng = np.random.default_rng(1)
    canvas = np.clip(rng.normal(130, 35, (800, 580, 1)), 0, 255).astype(np.uint8).repeat(3, 2)  # countertop
    canvas[40:760, 40:540] = dim
    out = clean_scan(canvas, upscale=1.0, style="bw")
    assert out.ndim == 2
    inner = out[100:700, 100:480]
    assert np.median(inner) == 255                            # paper is white
    assert (inner < 80).mean() > 0.01                         # text is dark
    assert (out[:30] == 255).mean() > 0.95                    # countertop is gone
    nat = cv2.cvtColor(clean_scan(canvas, upscale=1.0), cv2.COLOR_BGR2GRAY)  # natural: the paper's own tone
    inner = nat[100:700, 100:480]
    assert 225 <= np.median(inner) < 255                      # bright, but not blown out
    assert (inner < 80).mean() > 0.01


def test_pale_hand_is_skin_on_white_paper():
    img = np.full((600, 400, 3), 235, np.uint8)               # white paper filling the frame
    lab = np.uint8([[[205 * 255 // 255, 128 + 5, 128 + 10]]])
    skin = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]          # washed-out skin: a*~5, b*~10
    cv2.ellipse(img, (0, 300), (160, 60), 0, 0, 360, skin.tolist(), -1)  # a hand coming in from the side
    assert (skin_mask(img) > 0).mean() < 0.01                 # too pale for the strict colour test
    assert (skin_mask(img, pale=True)[300, 60:140] > 0).all()


def test_ffmpeg_reader_matches_opencv(tmp_path):
    frames, _ = _page_frames(2)
    p = _write(str(tmp_path / "v.mp4"), frames)
    src = VideoSource(p)
    if src.decoder == "opencv":
        pytest.skip("ffmpeg not available")
    idx = [0, 17, 40]
    ff = src.fetch(idx)
    cap = cv2.VideoCapture(p)
    for i in range(max(idx) + 1):
        ok, f = cap.read()
        if i in idx:
            assert np.abs(f.astype(int) - ff[i].astype(int)).mean() < 6
    steps = [i for i, _ in src.frames(3, width=200)]
    assert steps[:3] == [0, 3, 6]


# ---------------------------------------------------------------- bad files

def test_missing_file(tmp_path):
    with pytest.raises(VideoError):
        VideoSource(str(tmp_path / "nope.mp4"))


def test_empty_file(tmp_path):
    p = tmp_path / "empty.mp4"; p.write_bytes(b"")
    with pytest.raises(VideoError):
        VideoSource(str(p))


def test_not_a_video(tmp_path):
    p = tmp_path / "notes.mp4"; p.write_text("this is not a video at all\n" * 100)
    with pytest.raises(VideoError):
        VideoSource(str(p))


def test_truncated_file_still_yields_pages(tmp_path):
    frames, _ = _page_frames(2)
    p = _write(str(tmp_path / "v.mp4"), frames)
    data = open(p, "rb").read()
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(data[: int(len(data) * 0.7)])  # upload interrupted
    pages = process_video(str(cut), str(tmp_path / "out"), Config(ocr=False, pdf=False))
    assert len(pages) >= 1


# ------------------------------------------------------- odd capture habits

def test_single_still_page(tmp_path):
    img, _ = mtv.render_page(*mtv.TEXTS[0], 1)
    base = cv2.resize(mtv.place(img, [[110, 330], [970, 345], [985, 1600], [95, 1590]]), (540, 960))
    rng = np.random.default_rng(0)
    frames = [np.clip(base + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8) for _ in range(45)]
    p = _write(str(tmp_path / "still.mp4"), frames)
    pages = process_video(p, str(tmp_path / "out"), Config(ocr=False, pdf=False))
    assert len(pages) == 1


def test_never_pauses_does_not_crash(tmp_path):
    frames, _ = _page_frames(3, hold=0.1, flip=0.6)
    p = _write(str(tmp_path / "rush.mp4"), frames)
    pages = process_video(p, str(tmp_path / "out"), Config(ocr=False, pdf=False))
    assert len(pages) >= 1
    m = json.load(open(tmp_path / "out" / "manifest.json"))
    assert m["pages"]


def test_sideways_video_is_auto_rotated(tmp_path):
    frames, truths = _page_frames(1, hold=1.5, rot=cv2.ROTATE_90_COUNTERCLOCKWISE)
    p = _write(str(tmp_path / "side.mp4"), frames)
    pages = process_video(p, str(tmp_path / "out"), Config(pdf=False))
    assert pages and _sim(truths[0], pages[0].text) > 0.8


def test_high_fps_is_subsampled(tmp_path):
    frames, _ = _page_frames(2, fps=120, hold=0.8, flip=0.3)
    p = _write(str(tmp_path / "slowmo.mp4"), frames, fps=120)
    pages = process_video(p, str(tmp_path / "out"), Config(ocr=False, pdf=False))
    assert len(pages) == 2


# ------------------------------------------------------------- components

def test_yellowed_paper_is_not_skin():
    im = cv2.imread(os.path.join(DATA, "boston_cooking_a.jpg"))
    h, w = im.shape[:2]
    centre = im[h // 5: 4 * h // 5, w // 5: 4 * w // 5]
    assert (skin_mask(centre) > 0).mean() < 0.005


def test_finger_is_skin():
    img, _ = mtv.render_page(*mtv.TEXTS[0], 1)
    f = mtv.finger(mtv.place(img, [[110, 330], [970, 345], [985, 1600], [95, 1590]]), (-40, 900), (300, 860), 80)
    m = skin_mask(cv2.resize(f, (540, 960)))
    assert m[440:460, 0:130].mean() > 100


def test_dewarp_straightens_curled_page():
    img, _ = mtv.render_page(*mtv.TEXTS[1], 1)
    curled = mtv.curl(img, "left", 0.05)
    out, info = dewarp(curled)
    assert info["dewarped"]

    def curl_px(im):
        lines = find_lines(text_mask(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)))
        return np.mean([np.ptp(l[:, 1]) for l in lines])
    assert curl_px(out) < 0.6 * curl_px(curled)


def test_dewarp_leaves_blank_and_picture_pages_alone():
    blank = np.full((1200, 800, 3), 230, np.uint8)
    out, info = dewarp(blank)
    assert not info["dewarped"] and out is blank
    noise = np.random.default_rng(1).integers(0, 255, (900, 700, 3), dtype=np.uint8)
    out, info = dewarp(noise)
    assert out.shape == noise.shape


def test_reading_order_two_columns():
    def w(t, x, y):
        return Word(t, 0.9, np.float32([[x, y], [x + 80, y], [x + 80, y + 20], [x, y + 20]]))
    words = [w(f"L{i}", 10, 40 * i) for i in range(8)] + [w(f"R{i}", 400, 40 * i) for i in range(8)]
    assert words_to_text(words).split("\n")[:9] == [f"L{i}" for i in range(8)] + ["R0"]


# --------------------------------------------------------------- end to end

@pytest.mark.slow
def test_end_to_end(tmp_path):
    vid = str(tmp_path / "book.mp4")
    subprocess.run([sys.executable, os.path.join(HERE, "make_test_video.py"), vid, "--real-dir", DATA], check=True)
    truth = json.load(open(str(tmp_path / "book.truth.json")))
    pages = process_video(vid, str(tmp_path / "out"), Config())
    expected = sum(2 if t["spread_text"] else 1 for t in truth["pages"])
    assert len(pages) == expected
    texts = [p.text for p in pages]
    for t in truth["pages"]:
        if t["text"] and t["scenario"] != "finger_on_text":
            assert max(_sim(t["text"], x) for x in texts) > 0.95, t["name"]
    flagged = {f for p in pages for f in p.flags}
    assert "finger_may_cover_text" in flagged and "short_hold" in flagged
    assert os.path.exists(tmp_path / "out" / "book.pdf")
