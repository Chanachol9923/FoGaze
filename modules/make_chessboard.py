"""Generate a printable calibration chessboard.

Default matches ``StereoCalibrator`` / ``CameraCalibrator``: 9x6 *internal*
corners (i.e. a 10x7 grid of squares) at 25 mm per square.

Outputs two files into ``assets/``:
  * an SVG sized in real millimetres  -> print this for accurate squares
  * a PNG at 300 DPI                  -> quick on-screen preview

Print the SVG at **100% / "Actual size"** (no "fit to page" scaling) on
**A4 landscape**, then measure one square with a ruler to confirm it is 25 mm.
If your printer scaled it, edit SQUARE_MM in the calibrators to the measured
value instead.

    python3 modules/make_chessboard.py                 # 9x6 corners, 25 mm
    python3 modules/make_chessboard.py --cols 9 --rows 6 --square 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_svg(sq_cols: int, sq_rows: int, square_mm: float) -> str:
    w_mm = sq_cols * square_mm
    h_mm = sq_rows * square_mm
    rects = []
    for r in range(sq_rows):
        for c in range(sq_cols):
            if (r + c) % 2 == 0:
                x, y = c * square_mm, r * square_mm
                rects.append(
                    f'<rect x="{x:.3f}" y="{y:.3f}" '
                    f'width="{square_mm:.3f}" height="{square_mm:.3f}" fill="black"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w_mm:.3f}mm" height="{h_mm:.3f}mm" '
        f'viewBox="0 0 {w_mm:.3f} {h_mm:.3f}">\n'
        f'<rect width="{w_mm:.3f}" height="{h_mm:.3f}" fill="white"/>\n'
        + "\n".join(rects)
        + "\n</svg>\n")


def make_png(sq_cols: int, sq_rows: int, square_mm: float, dpi: int = 300) -> np.ndarray:
    px = int(round(square_mm / 25.4 * dpi))  # pixels per square
    img = np.full((sq_rows * px, sq_cols * px, 3), 255, np.uint8)
    for r in range(sq_rows):
        for c in range(sq_cols):
            if (r + c) % 2 == 0:
                img[r * px:(r + 1) * px, c * px:(c + 1) * px] = 0
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make a printable calibration chessboard")
    ap.add_argument("--cols", type=int, default=9,
                    help="internal corners across (default 9 -> 10 squares)")
    ap.add_argument("--rows", type=int, default=6,
                    help="internal corners down (default 6 -> 7 squares)")
    ap.add_argument("--square", type=float, default=25.0, help="square size in mm")
    ap.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    args = ap.parse_args()

    # Internal corners -> squares = corners + 1 in each direction.
    sq_cols, sq_rows = args.cols + 1, args.rows + 1

    out = Path(__file__).resolve().parent.parent / "assets"
    out.mkdir(exist_ok=True)
    stem = f"chessboard_{args.cols}x{args.rows}_{args.square:g}mm"

    svg_path = out / f"{stem}.svg"
    svg_path.write_text(make_svg(sq_cols, sq_rows, args.square))

    png = make_png(sq_cols, sq_rows, args.square, args.dpi)
    png_path = out / f"{stem}.png"
    cv2.imwrite(str(png_path), png)

    print(f"Board: {args.cols}x{args.rows} internal corners "
          f"({sq_cols}x{sq_rows} squares), {args.square:g} mm each")
    print(f"Physical size: {sq_cols*args.square:g} x {sq_rows*args.square:g} mm "
          f"(fits A4 landscape)")
    print(f"  SVG (print this @100%): {svg_path}")
    print(f"  PNG (preview)         : {png_path}  {png.shape[1]}x{png.shape[0]}px")
