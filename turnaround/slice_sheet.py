#!/usr/bin/env python
"""Slice a character turnaround sheet into RGBA views named for Hunyuan3D-2mv.

Figures are found left to right, and --views names them in that order. 2mv's labels describe where the
character faces in the image: `left` faces image-left, `right` faces image-right.

Every crop keeps the sheet's full height, so the views stay on the same ground and eye lines.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from rembg import new_session, remove
from scipy import ndimage


def find_figures(alpha, expected, min_gap=20):
    """Column spans of the opaque regions, keeping the `expected` widest ones in left-to-right order."""
    columns = np.flatnonzero((alpha > 128).sum(axis=0) > 3)
    if columns.size == 0:
        raise SystemExit("no foreground found on the sheet")
    spans = [(run[0], run[-1]) for run in np.split(columns, np.flatnonzero(np.diff(columns) > min_gap) + 1)]
    if len(spans) < expected:
        raise SystemExit(f"found {len(spans)} figures but --views names {expected}: {spans}")
    return sorted(sorted(spans, key=lambda s: s[1] - s[0])[-expected:])


def clean_matte(image, max_hole=1500):
    """Keep the figure's largest component and fill small interior holes. Large holes are real gaps, such as between arm and body."""
    alpha = np.array(image.getchannel("A"))
    solid = alpha > 128
    labels, count = ndimage.label(solid)
    if count == 0:
        raise SystemExit("segmentation found nothing in a figure crop")
    largest = np.argmax(ndimage.sum(solid, labels, range(1, count + 1))) + 1
    alpha[labels != largest] = 0

    interior = ndimage.binary_fill_holes(labels == largest) & (labels != largest)
    holes, count = ndimage.label(interior)
    for index in range(1, count + 1):
        hole = holes == index
        if hole.sum() <= max_hole:
            alpha[hole] = 255
    cleaned = image.copy()
    cleaned.putalpha(Image.fromarray(alpha))
    return cleaned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sheet")
    parser.add_argument("out_dir")
    parser.add_argument("--views", default="front,right,back", help="view names, in left-to-right sheet order")
    parser.add_argument("--mirror-side", action="store_true", help="mirror the one profile to fill in the opposite side")
    parser.add_argument("--rembg-model", default="birefnet-general")
    parser.add_argument("--pad", type=int, default=30, help="horizontal padding around each figure, in sheet pixels")
    args = parser.parse_args()

    views = args.views.split(",")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet = Image.open(args.sheet).convert("RGB")
    session = new_session(args.rembg_model)

    coarse = np.array(remove(sheet, session=session).getchannel("A"))
    spans = find_figures(coarse, len(views))

    results = {}
    for view, (x0, x1) in zip(views, spans):
        box = (max(0, x0 - args.pad), 0, min(sheet.width, x1 + args.pad + 1), sheet.height)
        # Segmenting one figure at a time gives the matte far more resolution than the whole-sheet pass.
        results[view] = clean_matte(remove(sheet.crop(box), session=session))
        print(f"{view}: columns {x0}-{x1}, alpha bbox {results[view].getchannel('A').point(lambda a: 255 if a > 128 else 0).getbbox()}")

    if args.mirror_side:
        opposite = {"left": "right", "right": "left"}
        side = next((v for v in results if v in opposite), None)
        if side is None:
            raise SystemExit("--mirror-side needs a left or right view")
        if opposite[side] in results:
            raise SystemExit("--mirror-side is redundant: both sides are already present")
        results[opposite[side]] = ImageOps.mirror(results[side])

    for view, image in results.items():
        image.save(out / f"{view}.png")
    print("wrote", ", ".join(sorted(results)), "to", out)


if __name__ == "__main__":
    main()
