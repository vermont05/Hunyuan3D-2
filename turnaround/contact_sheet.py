#!/usr/bin/env python
"""Render meshes from the four sides into one comparison image, a row per mesh.

Run with Wan2GP's venv, from the Wan2GP directory:
    cd ~/Wan2GP && .venv/bin/python ~/Hunyuan3D-2/turnaround/contact_sheet.py OUT.png MESH.glb [MESH.glb ...]
"""
import argparse
import os
import sys

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.getcwd())
from shared.mesh3d import load_mesh  # noqa: E402
from shared.mesh3d.render import render_views  # noqa: E402

ANGLES = [(0, 0), (90, 0), (180, 0), (270, 0), (35, 15)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("meshes", nargs="+")
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--mode", default="shaded", help="render mode, e.g. shaded, albedo")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sheet = Image.new("RGB", (args.size * len(ANGLES), args.size * len(args.meshes)), (235, 235, 235))
    for row, path in enumerate(args.meshes):
        mesh = load_mesh(path, device=device)
        views = render_views(mesh, ANGLES, args.size, args.mode, supersample=2).cpu()
        for column, view in enumerate(views):
            tile = Image.fromarray((view.numpy() * 255).round().astype("uint8"))
            sheet.paste(tile.convert("RGB"), (column * args.size, row * args.size), tile.getchannel("A") if tile.mode == "RGBA" else None)
        ImageDraw.Draw(sheet).text((6, row * args.size + 6), os.path.basename(path), fill=(0, 0, 0))
    sheet.save(args.output)


if __name__ == "__main__":
    main()
