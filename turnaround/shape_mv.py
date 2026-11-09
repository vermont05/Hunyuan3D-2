#!/usr/bin/env python
"""Hunyuan3D-2mv: RGBA views from slice_sheet.py to an untextured mesh."""
import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen.postprocessors import DegenerateFaceRemover, FaceReducer, FloaterRemover

VIEWS = ("front", "left", "back", "right")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views_dir")
    parser.add_argument("output", help="output .glb path")
    parser.add_argument("--views", default=None, help="comma-separated subset to use, e.g. front,back,right")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--octree", type=int, default=380)
    parser.add_argument("--chunks", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-faces", type=int, default=0, help="decimate to this many faces (0 keeps all)")
    args = parser.parse_args()

    names = args.views.split(",") if args.views else VIEWS
    paths = {name: Path(args.views_dir) / f"{name}.png" for name in names}
    images = {name: Image.open(path).convert("RGBA") for name, path in paths.items() if path.exists()}
    if not images:
        raise SystemExit(f"no view images found in {args.views_dir}")
    print("views:", ", ".join(images))

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv", variant="fp16"
    )
    start = time.time()
    mesh = pipeline(
        image=images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        octree_resolution=args.octree,
        num_chunks=args.chunks,
        generator=torch.manual_seed(args.seed),
        output_type="trimesh",
    )[0]
    mesh = DegenerateFaceRemover()(FloaterRemover()(mesh))
    if args.max_faces:
        mesh = FaceReducer()(mesh, max_facenum=args.max_faces)
    print(f"{time.time() - start:.1f}s, {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)


if __name__ == "__main__":
    main()
