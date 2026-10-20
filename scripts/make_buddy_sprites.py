"""Cut the 5x3 robot sprite sheet into transparent PNG animation strips (the checkerboard is baked into the JPEG).

Usage: python scripts/make_buddy_sprites.py assets/buddy_sheet.jpg static/buddy [preview.png]
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
im = np.asarray(Image.open(src).convert("RGB")).astype(int)
H, W, _ = im.shape
CHECK = np.array([[203, 206, 211], [229, 233, 236]])


def bg_like(px):
    d = np.min(np.abs(px[..., None, :] - CHECK).sum(-1), axis=-1)
    return d < 22


ROWS, COLS = 3, 5
ch, cw = H / ROWS, W / COLS
frames = []
for r in range(ROWS):
    for c in range(COLS):
        y0, y1, x0, x1 = int(r * ch), int((r + 1) * ch), int(c * cw), int((c + 1) * cw)
        cell = im[y0:y1, x0:x1]
        h, w, _ = cell.shape
        cand = bg_like(cell)
        bg = np.zeros((h, w), bool)
        q = deque()
        for x in range(w):
            for y in (0, h - 1):
                if cand[y, x]:
                    bg[y, x] = True; q.append((y, x))
        for y in range(h):
            for x in (0, w - 1):
                if cand[y, x] and not bg[y, x]:
                    bg[y, x] = True; q.append((y, x))
        while q:
            y, x = q.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not bg[ny, nx] and cand[ny, nx]:
                    bg[ny, nx] = True; q.append((ny, nx))
        # enclosed checkerboard patches (e.g. between an arm and the body): a region made of BOTH checker shades
        near = np.abs(cell[..., None, :] - CHECK).sum(-1)
        shade = np.argmin(near, axis=-1)
        seen = bg.copy()
        for sy in range(h):
            for sx in range(w):
                if seen[sy, sx] or not cand[sy, sx]:
                    continue
                comp, q = [], deque([(sy, sx)])
                seen[sy, sx] = True
                while q:
                    y, x = q.popleft()
                    comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and cand[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx))
                ys, xs = np.array(comp).T
                share = shade[ys, xs].mean()
                if len(comp) >= 40 and 0.12 <= share <= 0.88:
                    bg[ys, xs] = True
                    continue
                # a lone checker square cut off at the cell edge: flat colour, fills its bounding box
                box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
                if len(comp) >= 60 and len(comp) / box > 0.75 and cell[ys, xs].std(axis=0).max() < 6:
                    bg[ys, xs] = True
        # drop the floor shadow: low-saturation grey-blue blob in the bottom part of the cell
        sat = cell.max(-1) - cell.min(-1)
        lum = cell.mean(-1)
        shadow = (np.arange(h)[:, None] > h * 0.80) & (sat < 45) & (lum > 120) & (lum < 215)
        alpha = np.where(bg | shadow, 0, 255).astype(np.uint8)
        a = Image.fromarray(alpha).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.6))
        rgba = Image.fromarray(cell.astype(np.uint8)).convert("RGBA")
        rgba.putalpha(a)
        frames.append(rgba)

# common crop box across all frames so the character doesn't jump between frames
boxes = [f.getbbox() for f in frames]
x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes); x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
side = max(x1 - x0, y1 - y0)
SIZE = 220
for i, f in enumerate(frames):
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(f.crop((x0, y0, x1, y1)), ((side - (x1 - x0)) // 2, side - (y1 - y0)))
    frames[i] = canvas.resize((SIZE, SIZE), Image.LANCZOS)

STATES = {"idle": [0, 1, 2, 3, 4], "think": [5, 6, 7, 8], "aha": [9], "present": [10, 11, 12], "celebrate": [13, 14]}
for name, idx in STATES.items():
    strip = Image.new("RGBA", (SIZE * len(idx), SIZE), (0, 0, 0, 0))
    for j, k in enumerate(idx):
        strip.paste(frames[k], (j * SIZE, 0))
    strip.save(out / f"{name}.png", optimize=True)
frames[0].resize((96, 96), Image.LANCZOS).save(out / "avatar.png", optimize=True)
preview = Image.new("RGBA", (SIZE * 5, SIZE * 3), (40, 44, 52, 255))
for i, f in enumerate(frames):
    preview.alpha_composite(f, ((i % 5) * SIZE, (i // 5) * SIZE))
preview.convert("RGB").save(sys.argv[3] if len(sys.argv) > 3 else out / "preview.png")
print("frames", len(frames), "crop", (x0, y0, x1, y1))
