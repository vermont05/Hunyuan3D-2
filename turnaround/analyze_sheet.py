#!/usr/bin/env python
"""Find the figures on a turnaround sheet and work out which view each one is.

Writes figure_N.png (RGBA) and analysis.json to OUT_DIR. Figures may sit in a row or a grid. CLIP finds the back view,
silhouette symmetry finds the front and rejects turned poses, and the direction a profile faces comes from where its
feet point. A sheet can show more poses than 2mv accepts, so 3/4 poses and views beyond two sides are marked `skip`.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from rembg import new_session, remove
from scipy import ndimage
from transformers import CLIPModel, CLIPProcessor

from slice_sheet import clean_matte

CLIP_MODEL = "openai/clip-vit-large-patch14"
PROMPTS = {
    "front": "a full body character seen from the front, facing the camera, with the face visible",
    "side": "a full body character seen from the side in profile",
    "back": "a full body character seen from behind, showing the back of the head with no face visible",
    "other": "a full body character at a three-quarter angle",
}
MIN_FIGURE_AREA = 0.2  # of the largest figure; drops labels, props and stray matte
PAD = 30
MIN_BACK_SYMMETRY = 0.78  # a profile with a hidden face can score as a back; measured profiles reach 0.74, backs start at 0.82
MAX_SIDE_SYMMETRY = 0.85  # of the front's symmetry; measured profiles reach 0.75, a 3/4 pose 0.96


def find_figures(alpha):
    """Bounding boxes (x0, y0, x1, y1) of the figures, in reading order. Nearby pieces of one figure are merged first."""
    solid = alpha > 128
    gap = max(3, round(min(alpha.shape) * 0.012))
    labels, count = ndimage.label(ndimage.binary_dilation(solid, iterations=gap))
    if count == 0:
        raise SystemExit("No character found in this image.")
    areas = ndimage.sum(solid, labels, range(1, count + 1))
    boxes = []
    for index, slices in enumerate(ndimage.find_objects(labels)):
        if areas[index] >= MIN_FIGURE_AREA * areas.max():
            part = np.argwhere(solid[slices] & (labels[slices] == index + 1))
            y0, x0 = part.min(axis=0) + (slices[0].start, slices[1].start)
            y1, x1 = part.max(axis=0) + (slices[0].start, slices[1].start)
            boxes.append((int(x0), int(y0), int(x1) + 1, int(y1) + 1))
    # Boxes whose vertical spans overlap share a row; rows read top to bottom, each left to right.
    boxes.sort(key=lambda b: b[1])
    rows = []
    for box in boxes:
        row = next((r for r in rows if box[1] < (r[0][1] + r[0][3]) / 2), None)
        if row is None:
            rows.append([box])
        else:
            row.append(box)
    return [box for row in rows for box in sorted(row)]


def on_white_square(rgba):
    figure = rgba.crop(rgba.getchannel("A").getbbox())
    side = max(figure.size)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(figure, ((side - figure.width) // 2, (side - figure.height) // 2), figure.getchannel("A"))
    return square


def classify(figures):
    """Probabilities [figure, class] over PROMPTS."""
    device = "cpu"  # a generation may be using the GPU, and this takes only seconds on the CPU
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True).to(device).eval()
    inputs = processor(text=list(PROMPTS.values()), images=[on_white_square(f) for f in figures], return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        scores = model(**inputs).logits_per_image.softmax(dim=-1).cpu().numpy()
    return scores


def facing(rgba):
    """Positive when the feet point image-right, negative when image-left, as a fraction of figure height."""
    alpha = np.array(rgba.getchannel("A")) > 128
    rows = np.flatnonzero(alpha.any(axis=1))
    top, bottom = rows[0], rows[-1]
    height = bottom - top + 1

    def center(start, end):
        return np.flatnonzero(alpha[top + int(start * height):top + int(end * height)].any(axis=0))

    feet, legs = center(0.94, 1.0), center(0.72, 0.88)
    if feet.size == 0 or legs.size == 0:
        return 0.0
    return float(((feet[0] + feet[-1]) - (legs[0] + legs[-1])) / 2 / height)


def symmetry(rgba):
    """Overlap of a figure's silhouette with its mirror image about its centre of mass, from 0 to 1."""
    alpha = np.array(rgba.getchannel("A")) > 128
    center = np.nonzero(alpha)[1].mean()
    margin = int(abs(alpha.shape[1] - 1 - 2 * center)) + 1
    padded = np.pad(alpha, ((0, 0), (margin, margin)))
    mirrored = np.roll(padded[:, ::-1], int(round(2 * (center + margin) - (padded.shape[1] - 1))), axis=1)
    return float((padded & mirrored).sum() / (padded | mirrored).sum())


def assign(scores, directions, symmetries):
    """One label per figure. CLIP finds the back but ranks a 3/4 pose above a true front, so the front is the most symmetric figure left."""
    names = list(PROMPTS)
    front, side, back = (scores[:, names.index(n)] for n in ("front", "side", "back"))
    labels = ["skip"] * len(scores)
    remaining = list(range(len(scores)))
    back_index = max(remaining, key=lambda i: back[i] - side[i])
    if back[back_index] > max(front[back_index], side[back_index]) and symmetries[back_index] >= MIN_BACK_SYMMETRY:
        labels[back_index] = "back"
        remaining.remove(back_index)
    front_index = max(remaining, key=lambda i: symmetries[i])
    labels[front_index] = "front"
    remaining.remove(front_index)
    # A leftover nearly as symmetric as the front is a turned pose, which would mislead the shape step if it were called a profile.
    remaining = [i for i in remaining if symmetries[i] < MAX_SIDE_SYMMETRY * symmetries[front_index]]
    remaining.sort(key=lambda i: side[i] - scores[i, names.index("other")], reverse=True)
    sides = remaining[:2]
    if len(sides) == 2 and (directions[sides[0]] > 0) == (directions[sides[1]] > 0):
        # They appear to face the same way, so the clearer reading decides and the other takes the opposite side.
        clear, other = sorted(sides, key=lambda i: abs(directions[i]), reverse=True)
        labels[clear] = "right" if directions[clear] > 0 else "left"
        labels[other] = "left" if directions[clear] > 0 else "right"
    else:
        for i in sides:
            labels[i] = "right" if directions[i] > 0 else "left"
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sheet")
    parser.add_argument("out_dir")
    parser.add_argument("--rembg-model", default="birefnet-general")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        sheet = Image.open(args.sheet).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise SystemExit("That file could not be read as an image.")
    sheet.save(out / "sheet.png")
    session = new_session(args.rembg_model)
    boxes = find_figures(np.array(remove(sheet, session=session).getchannel("A")))
    if len(boxes) < 2:
        raise SystemExit("Found only one character. Use an image that shows at least two views, with a clear gap between them.")

    figures = []
    for x0, y0, x1, y1 in boxes:
        box = (max(0, x0 - PAD), max(0, y0 - PAD), min(sheet.width, x1 + PAD), min(sheet.height, y1 + PAD))
        figures.append(clean_matte(remove(sheet.crop(box), session=session)))

    scores = classify(figures)
    directions = [facing(f) for f in figures]
    symmetries = [symmetry(f) for f in figures]
    labels = assign(scores, directions, symmetries)

    result = []
    for index, (figure, label) in enumerate(zip(figures, labels)):
        figure.save(out / f"figure_{index}.png")
        result.append({"file": f"figure_{index}.png", "label": label, "box": boxes[index], "facing": round(directions[index], 3), "symmetry": round(symmetries[index], 3),
                       "scores": {name: round(float(s), 3) for name, s in zip(PROMPTS, scores[index])}})
    (out / "analysis.json").write_text(json.dumps({"figures": result}, indent=2))
    print(json.dumps([(r["label"], r["facing"], r["scores"]) for r in result]))


if __name__ == "__main__":
    main()
