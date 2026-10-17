"""Build a realistic flip-through test video with known ground truth.

Mixes real photos of curved, yellowed book pages (from the page-dewarp
project's examples) with rendered pages whose text is known, then simulates
what goes wrong in real phone footage:

* handheld shake, slow exposure drift, sensor noise, h264 compression
* page-turn motion with a hand sweeping across
* fingers resting on margins (moving mid-hold, or static over text)
* a too-short hold, a hesitation (hand twitch) in the middle of a hold
* an underexposed page, a two-page spread, and a corrupted file tail

Usage: python make_test_video.py OUT.mp4 [--real-dir DIR]
Writes OUT.mp4 and OUT.truth.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 30
W, H = 1080, 1920
rng = np.random.default_rng(7)


def _first_font(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no serif font found; tried {paths}")


FONT = _first_font("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                   "/Library/Fonts/Georgia.ttf", "C:/Windows/Fonts/georgia.ttf")
FONT_B = _first_font("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
                     "/Library/Fonts/Georgia Bold.ttf", "C:/Windows/Fonts/georgiab.ttf")

TEXTS = [
    ("On Sewing the Signatures",
     "Before any binding can begin, the folded sheets must be gathered in order and pierced along "
     "the fold. The binder works from the first signature to the last, drawing linen thread through "
     "each station and locking it with a kettle stitch at the head and tail. A loose stitch here will "
     "show itself years later as a page that wanders from the spine. Patience at the frame is repaid "
     "by a book that opens flat and closes square."),
    ("Rounding and Backing",
     "Once the glue has set, the spine is gently tapped into a curve with a light hammer. This rounding "
     "keeps the fore edge from creeping forward as the book is read. The text block is then clamped "
     "between backing boards and the outer signatures are coaxed over to form shoulders, against which "
     "the covers will later hinge. Too much force cracks the folds; too little leaves the joint weak."),
    ("Choosing the Covering",
     "Cloth is cheap and forgiving, paper is light and easily printed, and leather is durable when it "
     "is pared thin at the edges. Whatever is chosen, the grain should run parallel to the spine, or the "
     "boards will warp as the paste dries. Many old account books were covered in a coarse buckram that "
     "has outlasted the ink written inside them."),
    ("Caring for Old Volumes",
     "Keep books upright and snug on the shelf, away from radiators and damp walls. Never pull a volume "
     "out by the top of its spine; push the neighbours back and grasp it by the middle. A soft brush "
     "removes dust better than a cloth, and a folded strip of acid free paper makes a kinder bookmark "
     "than a pencil or a pressed flower."),
    ("Marbled Endpapers",
     "The marbler floats colours on a bath of thickened water, combs them into patterns, and lays a "
     "sheet of paper on the surface to lift the design. No two sheets are ever quite alike. Endpapers "
     "made this way were once a mark of a careful shop, and collectors still use their patterns to guess "
     "where and when a binding was made."),
    ("Repairing a Torn Leaf",
     "A clean tear can be mended with wheat starch paste and a narrow strip of thin tissue. Feather the "
     "edges of the tissue by tearing rather than cutting, so the repair blends into the page. Weight the "
     "leaf between blotters overnight. Pressure sensitive tape should never be used, as its adhesive "
     "yellows and stains the paper within a few years."),
]


def wrap(draw, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def render_page(title, body, num, size=(900, 1300), paper=(236, 226, 205)):
    img = Image.new("RGB", size, paper)
    d = ImageDraw.Draw(img)
    f, fb, fs = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT_B, 36), ImageFont.truetype(FONT, 24)
    m = 90
    d.text((size[0] // 2, m), title, font=fb, fill=(30, 25, 20), anchor="mt")
    y = m + 80
    lines = wrap(d, body, f, size[0] - 2 * m)
    for ln in lines:
        d.text((m, y), ln, font=f, fill=(35, 30, 25))
        y += 46
    d.text((size[0] // 2, size[1] - 70), str(num), font=fs, fill=(40, 35, 30), anchor="mt")
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # paper texture + mild ageing blotches
    tex = cv2.GaussianBlur(rng.normal(0, 6, arr.shape[:2]).astype(np.float32), (0, 0), 1.2)
    blot = cv2.GaussianBlur(rng.normal(0, 1, (size[1] // 40, size[0] // 40)).astype(np.float32), (0, 0), 2)
    blot = cv2.resize(blot, size) * 10
    arr = np.clip(arr.astype(np.float32) + (tex + blot)[..., None], 0, 255).astype(np.uint8)
    truth = title + "\n" + "\n".join(lines) + "\n" + str(num)
    return arr, truth


def curl(page, spine="left", amount=0.045):
    """Bend the page like an open book: lines dip near the spine; shade the gutter."""
    h, w = page.shape[:2]
    xs = np.arange(w, dtype=np.float32) / w
    t = xs if spine == "left" else 1 - xs
    bend = np.exp(-t * 6.0)                      # strong near spine
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ny = (yy / h - 0.5) * 2
    dy = amount * h * bend[None, :] * (0.4 + 0.6 * np.abs(ny)) * np.sign(ny + 1e-6) * -1
    dy += 0.012 * h * np.sin(np.pi * xs)[None, :]  # gentle global sag
    mapx = xx
    mapy = (yy + dy).astype(np.float32)
    out = cv2.remap(page, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    shade = 1 - 0.45 * bend
    return (out.astype(np.float32) * shade[None, :, None]).clip(0, 255).astype(np.uint8)


def table(h=H, w=W):
    base = np.zeros((h, w, 3), np.float32) + (52, 70, 92)
    grain = cv2.resize(rng.normal(0, 1, (h // 8, 6)).astype(np.float32), (w, h)) * 14
    noise = cv2.GaussianBlur(rng.normal(0, 8, (h, w)).astype(np.float32), (0, 0), 2)
    return np.clip(base + (grain + noise)[..., None], 0, 255).astype(np.uint8)


TABLE = table()


def place(page, quad):
    """Warp page onto the table at quad (tl,tr,br,bl)."""
    h, w = page.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(quad))
    frame = TABLE.copy()
    warped = cv2.warpPerspective(page, M, (W, H))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    # soft drop shadow
    sh = cv2.GaussianBlur(cv2.warpAffine(mask, np.float32([[1, 0, 14], [0, 1, 18]]), (W, H)), (0, 0), 12)
    frame = (frame * (1 - 0.5 * sh[..., None] / 255.0)).astype(np.uint8)
    frame[mask > 0] = warped[mask > 0]
    return frame


def finger(frame, base, tip, width=70, rng_=None):
    """Draw a shaded finger from base (off-page) to tip with a fingernail."""
    base, tip = np.float32(base), np.float32(tip)
    v = tip - base; L = np.linalg.norm(v); u = v / L; n = np.float32([-u[1], u[0]])
    over = np.zeros(frame.shape[:2], np.float32)
    pts = np.int32([base + n * width / 2, tip + n * width * 0.45, tip - n * width * 0.45, base - n * width / 2])
    cv2.fillConvexPoly(over, pts, 1.0)
    cv2.circle(over, tuple(np.int32(tip)), int(width * 0.45), 1.0, -1)
    over = cv2.GaussianBlur(over, (0, 0), 2)
    skin = np.float32([118, 148, 205])  # BGR
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]].astype(np.float32)
    side = ((xx - base[0]) * n[0] + (yy - base[1]) * n[1]) / (width / 2)
    shade = (1 - 0.35 * np.clip(side, -1, 1) ** 2)[..., None]
    col = skin * shade
    out = frame.astype(np.float32) * (1 - over[..., None]) + col * over[..., None]
    nail = tip - u * width * 0.25
    cv2.ellipse(out, (tuple(int(c) for c in nail), (int(width * 0.32), int(width * 0.45)),
                      float(np.degrees(np.arctan2(u[1], u[0])) + 90)), (175, 190, 225), -1)
    return np.clip(out, 0, 255).astype(np.uint8)


class Camera:
    def __init__(self):
        self.p = np.zeros(4); self.v = np.zeros(4); self.expo = 1.0

    def step(self, shake=1.0):
        self.v = 0.85 * self.v + rng.normal(0, [0.8, 0.8, 0.03, 0.0008]) * shake
        self.p = 0.97 * self.p + self.v
        self.expo = float(np.clip(self.expo + rng.normal(0, 0.004), 0.88, 1.08))

    def apply(self, frame, gain=1.0):
        tx, ty, rot, sc = self.p
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, 1 + sc)
        M[:, 2] += (tx, ty)
        f = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        f = f.astype(np.float32) * self.expo * gain + rng.normal(0, 3.0, f.shape)
        return np.clip(f, 0, 255).astype(np.uint8)


def real_page(path):
    im = cv2.imread(path)
    s = min((W - 60) / im.shape[1], (H - 300) / im.shape[0])
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--real-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    a = ap.parse_args()

    pages = []  # (image, quad, kind, truth, extras)
    q = lambda dx=0, dy=0, k=0: [[110 + dx + k, 330 + dy], [970 + dx - k, 345 + dy], [985 + dx, 1600 + dy], [95 + dx, 1590 + dy]]
    for i, (t, b) in enumerate(TEXTS):
        img, tr = render_page(t, b, 101 + i)
        pages.append(dict(img=curl(img, "left" if i % 2 else "right"), quad=q(k=6 * (i % 3)), truth=tr, name=f"synthetic_{i}"))
    real = {}
    for nm in ["boston_cooking_a", "boston_cooking_b", "finnish_cooking_a"]:
        p = os.path.join(a.real_dir, nm + ".jpg")
        if os.path.exists(p):
            real[nm] = real_page(p)

    # spread: two rendered pages side by side, curled toward a shaded gutter
    l, tl_ = render_page("Tools of the Trade", TEXTS[0][1][:330], 210, size=(700, 1000))
    r, tr_ = render_page("The Bone Folder", TEXTS[1][1][:330], 211, size=(700, 1000))
    spread = np.hstack([curl(l, "right", 0.03), curl(r, "left", 0.03)])

    # (page, hold seconds, scenario)
    seq = [
        (pages[0], 1.8, "fingers_move"),        # fingers on margin, shift halfway -> composited out
        (pages[1], 1.6, "finger_on_text"),      # static finger over a text line -> flagged
        ("boston_cooking_a", 1.6, "plain"),     # real yellowed curved page
        ("boston_cooking_b", 1.6, "plain"),
        (pages[2], 0.25, "too_short"),          # flipped past almost immediately
        (pages[3], 2.4, "hesitate"),            # hand twitches mid-hold -> two segments, one page
        ("finnish_cooking_a", 1.5, "plain"),
        (pages[4], 1.6, "dark"),                # underexposed
        ("spread", 1.8, "plain"),               # two-page spread
        (pages[5], 1.6, "plain"),
    ]
    cam = Camera()
    tmp = tempfile.mkdtemp()
    raw = os.path.join(tmp, "raw.avi")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    truth = {"pages": [], "fps": FPS}
    prev = None
    t = 0

    def write(fr):
        nonlocal t
        vw.write(fr); t += 1

    def compose(item):
        if isinstance(item, dict):
            return place(item["img"], item["quad"])
        if item == "spread":
            return place(spread, [[60, 560], [1020, 575], [1030, 1270], [50, 1260]])
        f = TABLE.copy(); im = real[item]
        y0, x0 = (H - im.shape[0]) // 2, (W - im.shape[1]) // 2
        f[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
        return f

    for item, hold, scen in seq:
        if isinstance(item, str) and item not in ("spread",) and item not in real:
            continue
        frame = compose(item)
        # --- page turn: outgoing page squeezes toward spine, hand sweeps across
        if prev is not None:
            nT = 14
            for k in range(nT):
                a_ = (k + 1) / nT
                f = prev.copy() if a_ < 0.5 else frame.copy()
                cut = int(W * (1 - a_))
                f[:, :cut] = cv2.resize(prev, (max(1, cut), H))[:, :cut] if a_ < 1 else f[:, :cut]
                f = cv2.blur(f, (25, 1))
                hx = int(W * (1.1 - 1.2 * a_))
                f = finger(f, (hx + 300, H + 50), (hx, int(H * 0.55)), 120)
                cam.step(4.0)
                write(cam.apply(f))
        n = max(1, int(round(hold * FPS)))
        gain = 0.42 if scen == "dark" else 1.0
        for k in range(n):
            f = frame.copy()
            if scen == "fingers_move":
                yb = 1350 if k < n // 2 else 980
                f = finger(f, (-40, yb + 60), (150, yb), 78)
                f = finger(f, (W + 40, yb + 140), (W - 140, yb + 90), 78)
            elif scen == "finger_on_text":
                f = finger(f, (-40, 820), (330, 760), 80)   # reaches across text lines
            else:
                f = finger(f, (-40, 1500), (125, 1470), 74)  # thumb on the bottom-left margin
            if scen == "hesitate" and abs(k - n // 2) < 4:   # brief twitch
                cam.step(9.0)
            else:
                cam.step(0.5)
            write(cam.apply(f, gain))
        name = item if isinstance(item, str) else item["name"]
        truth["pages"].append({"name": name, "scenario": scen, "hold_s": hold,
                               "text": item["truth"] if isinstance(item, dict) else None,
                               "spread_text": [tl_, tr_] if item == "spread" else None})
        prev = frame
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-c:v", "libx264", "-preset", "medium",
                    "-crf", "26", "-pix_fmt", "yuv420p", a.out], check=True)
    # simulate a truncated/corrupted upload: garbage appended at the end
    with open(a.out, "ab") as fh:
        fh.write(rng.integers(0, 256, 200_000, dtype=np.uint8).tobytes())
    with open(os.path.splitext(a.out)[0] + ".truth.json", "w") as fh:
        json.dump(truth, fh, indent=1)
    print(f"wrote {a.out}: {t} frames, {t / FPS:.1f}s, {len(truth['pages'])} distinct pages")


if __name__ == "__main__":
    main()
