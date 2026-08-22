"""Make the repo social preview from Logo.jpg.

The background is a JPEG off-white (sampled luminance ~224-241, slight
color cast), so keying works in three steps:
  1. flood-fill from the borders through bright, low-saturation pixels
  2. choke that mask 1px inward to cut the transition band
  3. unmix remaining white fringe on a thin ring around the mask
     (GIMP-style color-to-alpha against white, ring only so light
     colors inside the logo are untouched)

Output canvas is GitHub's recommended social preview size: 1280x640 (2:1).
"""
from collections import deque

from PIL import Image, ImageChops, ImageFilter

SRC, DST = "Logo.jpg", "social_preview.png"
CHECK = "social_preview_check.png"
W, H = 1280, 640
LUM_T, SPREAD_T = 218, 26  # background: bright and low-saturation
CHOKE, RING = 1, 2  # px to cut inward, px of fringe to unmix

im = Image.open(SRC).convert("RGB")
w, h = im.size
r, g, b = im.split()
lum = Image.merge("RGB", (r, g, b)).convert("L")
mx = ImageChops.lighter(ImageChops.lighter(r, g), b)
mn = ImageChops.darker(ImageChops.darker(r, g), b)
spread = ImageChops.subtract(mx, mn)

elig = ImageChops.multiply(
    lum.point(lambda p: 255 if p >= LUM_T else 0),
    spread.point(lambda p: 255 if p <= SPREAD_T else 0),
)

# Flood fill eligibility from the borders
mask = elig.load()
visited = bytearray(w * h)
q = deque()
for x in range(w):
    q += [(x, 0), (x, h - 1)]
for y in range(h):
    q += [(0, y), (w - 1, y)]
while q:
    x, y = q.popleft()
    if 0 <= x < w and 0 <= y < h:
        i = y * w + x
        if not visited[i] and mask[x, y]:
            visited[i] = 1
            q += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

bg = Image.frombytes("L", (w, h), bytes(255 if v else 0 for v in visited))
bg = bg.filter(ImageFilter.MaxFilter(2 * CHOKE + 1))  # eat transition band
ring = ImageChops.subtract(
    bg.filter(ImageFilter.MaxFilter(2 * RING + 1)), bg
).point(lambda p: 255 if p else 0)

# Unmix white on the fringe ring; bg becomes fully transparent
src = im.load()
ring_px = ring.load()
bg_px = bg.load()
out = Image.new("RGBA", (w, h))
dst = out.load()
for y in range(h):
    for x in range(w):
        cr, cg, cb = src[x, y]
        if bg_px[x, y]:
            dst[x, y] = (0, 0, 0, 0)
        elif ring_px[x, y]:
            a = max(255 - cr, 255 - cg, 255 - cb)
            if a <= 8:
                dst[x, y] = (0, 0, 0, 0)
            else:
                inv = 255 - a
                dst[x, y] = (
                    max(0, min(255, round((cr - inv) * 255 / a))),
                    max(0, min(255, round((cg - inv) * 255 / a))),
                    max(0, min(255, round((cb - inv) * 255 / a))),
                    a,
                )
        else:
            dst[x, y] = (cr, cg, cb, 255)

# Feather the alpha edge a hair so it isn't jagged
out.putalpha(out.getchannel("A").filter(ImageFilter.GaussianBlur(0.7)))

# Trim, scale to fit with margin, center on the 1280x640 canvas
logo = out.crop(out.getbbox())
target_h = H - 80
scale = min(target_h / logo.height, (W - 80) / logo.width)
logo = logo.resize(
    (round(logo.width * scale), round(logo.height * scale)), Image.LANCZOS
)
canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
canvas.paste(logo, ((W - logo.width) // 2, (H - logo.height) // 2), logo)
canvas.save(DST)

# Diagnostic: composite onto mid-gray so any white fringe is obvious
canvas.convert("RGB").copy = None
check = Image.new("RGB", (W, H), (120, 120, 120))
check.paste(canvas, (0, 0), canvas)
check.save(CHECK)

a = canvas.getchannel("A")
print(f"saved {DST} {canvas.size}, logo {logo.size}")
print("corners:", [a.getpixel(p) for p in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)]])
print("left-mid / right-mid:", a.getpixel((100, H // 2)), a.getpixel((W - 100, H // 2)))
