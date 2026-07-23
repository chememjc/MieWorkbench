#!/usr/bin/env python
# =============================================================================
# gen_usaf_target.py — generate the shipped USAF-1951-STYLE resolution
# target bitmap for the extended image-emitting source (samples-instruments
# round). Interpreter: /home3/optics/env/bin/python (numpy + matplotlib).
#
# NOT a licensed reproduction of the MIL-STD-150A artwork — a public-domain
# STYLE-ALIKE: three-bar elements (horizontal + vertical pairs) in
# 2x-descending groups around the center, on a bright-emits convention
# (white bars on black = the bars EMIT light, like a backlit chrome-on-
# glass target). Regenerate with:
#   /home3/optics/env/bin/python scripts/tools/gen_usaf_target.py
# Writes opticalproperties/image/usaf_style_target.png (512x512, 8-bit grey).
# =============================================================================
import numpy as np
from pathlib import Path


def three_bars(img, x0, y0, w, h, horizontal):
    """Three bright bars of width w (pitch 2w) spanning h, top-left (x0,y0).
    Bar width == gap width (USAF convention: bar length = 5x width)."""
    for k in range(3):
        if horizontal:
            y = y0 + 2 * k * w
            img[y:y + w, x0:x0 + h] = 1.0
        else:
            x = x0 + 2 * k * w
            img[y0:y0 + h, x:x + w] = 1.0


def build(size=512):
    img = np.zeros((size, size))
    # groups of 2x-descending bar widths, laid out in an inward spiral of
    # quadrant blocks like the real target's group pairs
    layouts = [
        (32, 40, 40),      # (bar_w, x, y) coarsest group, top-left
        (16, 40, 300),
        (8, 300, 40),
        (4, 300, 200),
        (2, 300, 300),
        (1, 300, 380),
    ]
    for w, x, y in layouts:
        h = 5 * w
        three_bars(img, x, y, w, h, horizontal=False)
        gap = 2 * w
        three_bars(img, x + h + gap, y, w, h, horizontal=True)
    # central bright square (power/uniformity patch) + an asymmetric corner
    # mark so image orientation is testable end-to-end
    img[236:276, 236:276] = 1.0
    img[16:32, 16:64] = 1.0     # top-left orientation bar
    return img


def main():
    out = Path(__file__).resolve().parents[2] / "opticalproperties" \
        / "image" / "usaf_style_target.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img = build()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    mpimg.imsave(str(out), img, cmap="gray", vmin=0.0, vmax=1.0)
    print("wrote %s (%.1f%% bright)" % (out, 100.0 * (img > 0.5).mean()))


if __name__ == "__main__":
    main()
