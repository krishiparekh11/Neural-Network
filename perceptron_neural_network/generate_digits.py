"""
Generates 20x20 grayscale "handwritten" digits 0-9.

Each digit is drawn as pen strokes; every image gets a different random
slant, size, position, stroke thickness and wobble of the stroke points,
so no two images are the same - like writing the digit several times by hand.

    data/train/<digit>/<digit>_<k>.png    10 labeled images per digit (100 total)
    data/test/test_<nn>.png               unlabeled test images (random order)
    data/test_answer_key.csv              true labels of the test images (for checking only)
    data/train_preview.png, data/test_preview.png

Usage:  python generate_digits.py [--per-digit 10] [--test-per-digit 3] [--seed 7]
You can replace any of these images with scans of your own handwriting.
"""
import argparse
import csv
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HI = 80        # draw at 4x resolution, then shrink to 20x20 (smooth, anti-aliased)
OUT = 20


def arc(cx, cy, rx, ry, t0, t1, n=24):
    """Points on an ellipse; angles in degrees, y axis points DOWN (270 = top)."""
    t = np.radians(np.linspace(t0, t1, n))
    return list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t)))


def digit_strokes(d, rng):
    """Stroke list for digit d in a unit box (0..1). Small random shape changes included."""
    j = lambda s=0.03: rng.uniform(-s, s)
    if d == 0:
        return [arc(0.5, 0.5, 0.21 + j(), 0.35 + j(), 0, 360 + rng.uniform(0, 15), 40)]
    if d == 1:
        top = (0.52 + j(), 0.12 + j())
        s = [[top, (0.5 + j(), 0.88 + j())]]
        if rng.random() < 0.7:                              # flag on top
            s.append([(0.37 + j(), 0.27 + j()), top])
        if rng.random() < 0.3:                              # base line
            s.append([(0.38, 0.88), (0.64, 0.88)])
        return s
    if d == 2:
        top = arc(0.5, 0.33 + j(), 0.2 + j(), 0.19 + j(), 185, 395 + rng.uniform(-10, 10))
        return [top + [(0.27 + j(), 0.87 + j()), (0.76 + j(), 0.87 + j())]]
    if d == 3:
        up = arc(0.48, 0.31 + j(), 0.18 + j(), 0.17, 200, 450)
        lo = arc(0.48, 0.67 + j(), 0.21 + j(), 0.2, 270, 520)
        return [up + lo]
    if d == 4:
        x = 0.62 + j()
        return [[(0.26 + j(), 0.62 + j()), (x - 0.02, 0.12 + j())],
                [(0.22 + j(), 0.63 + j()), (0.79 + j(), 0.63 + j())],
                [(x, 0.3 + j()), (x + j(), 0.88 + j())]]
    if d == 5:
        body = arc(0.49, 0.64 + j(), 0.22 + j(), 0.22, 210, 510)
        return [[(0.72 + j(), 0.13 + j()), (0.33 + j(), 0.13 + j()), body[0]] + body]
    if d == 6:
        loop = arc(0.5 + j(), 0.67, 0.2 + j(), 0.19, 180, 540, 36)
        return [[(0.67 + j(), 0.13 + j()), (0.46, 0.25 + j()), (0.33 + j(), 0.45), loop[0]] + loop]
    if d == 7:
        s = [[(0.25 + j(), 0.14 + j()), (0.76 + j(), 0.13 + j()), (0.43 + j(0.05), 0.88 + j())]]
        if rng.random() < 0.3:                              # crossed 7
            s.append([(0.42, 0.5), (0.7, 0.5)])
        return s
    if d == 8:
        return [arc(0.5 + j(), 0.3, 0.17 + j(), 0.17, 0, 360, 30),
                arc(0.5 + j(), 0.67, 0.21 + j(), 0.2, 0, 360, 30)]
    if d == 9:
        cx, cy, r = 0.5 + j(), 0.33 + j(), 0.19 + j()
        s = [arc(cx, cy, r, 0.18, 0, 360, 30), [(cx + r, cy), (0.65 + j(), 0.88 + j())]]
        return s
    raise ValueError(d)


def render_digit(d, rng):
    strokes = digit_strokes(d, rng)

    # random "handwriting" transform: slant, rotation, size, position
    angle = np.radians(rng.uniform(-12, 12))
    shear = rng.uniform(-0.25, 0.15)
    scale = rng.uniform(0.8, 1.02)
    sx, sy = scale * rng.uniform(0.9, 1.1), scale
    tx, ty = rng.uniform(-0.06, 0.06), rng.uniform(-0.05, 0.05)
    c, s = np.cos(angle), np.sin(angle)

    def tf(p):
        x, y = p[0] - 0.5, p[1] - 0.5
        x = x + shear * y                                  # slant
        x, y = x * sx, y * sy
        x, y = c * x - s * y, s * x + c * y                # rotate
        x, y = x + 0.5 + tx, y + 0.5 + ty
        x += rng.normal(0, 0.008); y += rng.normal(0, 0.008)   # hand wobble
        return (x * HI, y * HI)

    img = Image.new("L", (HI, HI), 0)
    draw = ImageDraw.Draw(img)
    width = int(rng.integers(6, 11))                       # pen thickness
    ink = int(rng.integers(200, 256))                      # pen pressure
    for stroke in strokes:
        pts = [tf(p) for p in stroke]
        draw.line(pts, fill=ink, width=width, joint="curve")
        r = width / 2                                      # round pen tips
        for x, y in (pts[0], pts[-1]):
            draw.ellipse([x - r, y - r, x + r, y + r], fill=ink)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    return img.resize((OUT, OUT), Image.LANCZOS)


def preview(images, cols, path, scale=4):
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("L", (cols * (OUT + 2) * scale, rows * (OUT + 2) * scale), 60)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        sheet.paste(im.resize((OUT * scale, OUT * scale), Image.NEAREST),
                    (c * (OUT + 2) * scale, r * (OUT + 2) * scale))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--per-digit", type=int, default=10)
    ap.add_argument("--test-per-digit", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    train_imgs = []
    for d in range(10):
        folder = os.path.join(a.out, "train", str(d))
        os.makedirs(folder, exist_ok=True)
        for k in range(a.per_digit):
            im = render_digit(d, rng)
            im.save(os.path.join(folder, f"{d}_{k:02d}.png"))
            train_imgs.append(im)
    preview(train_imgs, a.per_digit, os.path.join(a.out, "train_preview.png"))

    # test set: new random variations, shuffled, filenames do NOT reveal the digit
    test_rng = np.random.default_rng(a.seed + 1000)
    digits = np.repeat(np.arange(10), a.test_per_digit)
    test_rng.shuffle(digits)
    folder = os.path.join(a.out, "test")
    os.makedirs(folder, exist_ok=True)
    test_imgs = []
    with open(os.path.join(a.out, "test_answer_key.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "label"])
        for i, d in enumerate(digits):
            im = render_digit(int(d), test_rng)
            name = f"test_{i + 1:02d}.png"
            im.save(os.path.join(folder, name))
            w.writerow([name, int(d)])
            test_imgs.append(im)
    preview(test_imgs, 10, os.path.join(a.out, "test_preview.png"))
    print(f"Saved {len(train_imgs)} training and {len(test_imgs)} test images to '{a.out}/'")


if __name__ == "__main__":
    main()
