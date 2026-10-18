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
from piffler.dewarp import deshear, dewarp, find_lines, text_mask  # noqa: E402
from piffler.hands import occluder_mask, skin_mask  # noqa: E402
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


@pytest.mark.parametrize("page_h", [420, 900, 1700])
def test_page_found_by_text_at_any_size(page_h):
    """Small book far away, or a big book filling the frame: found by its text either way."""
    rng = np.random.default_rng(5)
    h, w = 1920, 1080
    scene = np.clip(rng.normal(140, 35, (h, w, 1)), 0, 255).astype(np.uint8).repeat(3, 2)
    page, _ = mtv.render_page(*mtv.TEXTS[2], 9)
    pw = min(int(page_h * 0.6), w - 40)
    page = cv2.resize(page, (pw, page_h), interpolation=cv2.INTER_AREA if page_h < 1500 else cv2.INTER_CUBIC)
    y0, x0 = (h - page_h) // 2, (w - pw) // 2
    scene[y0:y0 + page_h, x0:x0 + pw] = page
    if page_h <= 600:  # a bright, text-free decoy near a small page (not touching it)
        cv2.circle(scene, (w // 2, 260), 220, (240, 241, 238), -1)
    quad, _, flags = detect_page(scene)
    assert "page_by_brightness" not in flags
    qx0, qy0 = quad.min(0)
    qx1, qy1 = quad.max(0)
    iou_w = max(0, min(qx1, x0 + pw) - max(qx0, x0)) / max(qx1 - qx0, pw)
    iou_h = max(0, min(qy1, y0 + page_h) - max(qy0, y0)) / max(qy1 - qy0, page_h)
    assert iou_w > 0.85 and iou_h > 0.85


@pytest.mark.parametrize("f", [0.6, 1.0, 1.8])
def test_clean_scan_at_any_print_size(f):
    page, _ = mtv.render_page(*mtv.TEXTS[0], 13)
    page = cv2.resize(page, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    dim = (page.astype(np.float32) * 0.8 + 25).astype(np.uint8)
    out = cv2.cvtColor(clean_scan(dim, upscale=1.0), cv2.COLOR_BGR2GRAY)
    h, w = out.shape
    inner = out[h // 5: 4 * h // 5, w // 6: 5 * w // 6]
    assert 225 <= np.median(inner) < 255   # paper bright, in its own tone
    assert (inner < 90).mean() > 0.005      # text still dark at this print size


def test_occluder_mask_finds_any_colour():
    page, _ = mtv.render_page(*mtv.TEXTS[1], 21)
    page = cv2.resize(page, (500, 720))
    rng = np.random.default_rng(2)
    frames = [np.clip(page + rng.normal(0, 3, page.shape), 0, 255).astype(np.uint8) for _ in range(5)]
    covered = frames[0].copy()
    cv2.rectangle(covered, (0, 300), (180, 420), (70, 90, 60), -1)  # a dark green sleeve: no skin colour
    region = np.full(page.shape[:2], 255, np.uint8)
    m = occluder_mask(covered, frames[1:], region)
    assert (m[320:400, 20:160] > 0).mean() > 0.8
    assert (m[:250] > 0).mean() < 0.01   # the rest of the page is left alone
    assert (occluder_mask(frames[0], frames[1:], region) > 0).mean() < 0.01


def test_location_prior_turns_off_for_handheld_video():
    from piffler.pipeline import _PagePrior
    q = np.float32([[0, 0], [100, 0], [100, 140], [0, 140]])
    stand, held = _PagePrior(), _PagePrior()
    for d in (0, 2, -1, 3):
        stand.add(q + d)
    for d in (0, 40, -35, 25):
        held.add(q + np.float32([d, d * 0.5]))
    assert stand.get((200, 200)) is not None
    assert held.get((200, 200)) is None


def test_facing_page_strip_is_dropped():
    from piffler.ocr import Word
    from piffler.pipeline import _drop_facing_column

    def page(stray_x=None, left_page=False):
        ws = []
        for k in range(20):
            y = 100 + k * 40
            if stray_x is not None:  # the facing page's line ends, past an empty margin
                ws.append(Word("xx", 0.9, np.float32([[stray_x, y], [stray_x + 25, y], [stray_x + 25, y + 20], [stray_x, y + 20]])))
            for j in range(5):
                x0 = (60 if left_page else 120) + j * 90
                ws.append(Word("mot", 0.9, np.float32([[x0, y], [x0 + 70, y], [x0 + 70, y + 20], [x0, y + 20]])))
        return ws

    kept, cut = _drop_facing_column(page(5), 600, gutter_right=False)
    assert len(kept) == 100 and 30 < cut < 120
    kept, cut = _drop_facing_column(page(560, left_page=True), 600, gutter_right=True)
    assert len(kept) == 100 and 480 < cut < 560
    assert _drop_facing_column(page(), 600, gutter_right=False)[1] is None
    blank_with_strays = [w for w in page(560, left_page=True) if w.text == "xx"][:4]  # a blank page
    kept, cut = _drop_facing_column(blank_with_strays, 600, gutter_right=True)
    assert kept == [] and 500 < cut < 560
    assert _drop_facing_column(page(left_page=True), 600, gutter_right=True)[1] is None


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


def _text_page(n_lines=30, w=900, h=1300, seed=4):
    """A full page of left-aligned lines (ragged right), like a novel's page."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 236, np.uint8)
    words = "le golf drouot etait un lieu mythique pour toute une generation de musiciens".split()
    for k in range(n_lines):
        y = 90 + k * 38
        x = 80 + (40 if k % 9 == 0 else 0)  # paragraph indents
        line = " ".join(rng.choice(words, 9))
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 2, cv2.LINE_AA)
    return img


def _column_lean(img):
    from piffler.dewarp import _edge_curve
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tm = text_mask(g)
    ys, starts = [], []
    for ln in find_lines(tm):
        y = int(np.median(ln[:, 1]))
        xs = np.nonzero(tm[max(0, y - 8):y + 9].any(0))[0]
        if len(xs) > 10:
            ys.append(y); starts.append(xs.min())
    c, _, _ = _edge_curve(np.float32(starts), np.float32(ys), deg=1)
    return abs(np.degrees(np.arctan(c[0])))


def test_deshear_stands_leaning_column_upright():
    page = _text_page()
    h, w = page.shape[:2]
    a = np.tan(np.radians(4.0))  # the column leans 4 degrees
    sheared = cv2.warpAffine(page, np.float32([[1, a, -a * h / 2], [0, 1, 0]]), (w, h),
                             borderMode=cv2.BORDER_REPLICATE)
    out, info = deshear(sheared)
    assert _column_lean(sheared) > 3.5
    assert _column_lean(out) < 0.5, info
    assert _column_lean(deshear(page)[0]) < 0.5            # an upright page stays upright
    blank = np.full((900, 600, 3), 235, np.uint8)
    assert deshear(blank)[1]["deshear_px"] == 0.0


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
