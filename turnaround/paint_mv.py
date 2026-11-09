#!/usr/bin/env python
"""Hunyuan3D 2.1 Paint PBR from several turnaround views, using Wan2GP's int8 model without modifying Wan2GP.

Run with Wan2GP's venv, from the Wan2GP directory:
    cd ~/Wan2GP && .venv/bin/python ~/Hunyuan3D-2/turnaround/paint_mv.py VIEWS_DIR MESH.glb OUT.glb

Stock 2.1 Paint takes one reference image, so everything out of its view is invented. This adds:
  --references  feeds every listed view into reference attention (its features are concatenated).
  --project     replaces the generated albedo of the front and back cameras with the real sheet pixels, fitted to the
                mesh silhouette, and bakes them with priority. Side views stay as references only.
  --hybrid      keeps the generated paint and adds those sheet pixels only where the surface faces the camera, so the
                drawing's detail stays while curved-away surfaces, which would stretch it, stay generated.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.getcwd())
from mmgp import offload  # noqa: E402

from models.hunyuan3d import paint_pipeline  # noqa: E402
from shared.attention import get_default_attention_mode  # noqa: E402
from shared.mesh3d import Mesh, load_mesh, mesh_to_glb  # noqa: E402
from shared.utils.default_device import call_with_default_device  # noqa: E402

MODEL_TYPE = "hunyuan3d_21_paint"
PROJECTED_CAMERAS = {0: "front", 2: "back"}  # select_views always starts with the cameras at azimuths 0, 90, 180, 270
# The front and back drawings must own every texel they can reach. Blended or wider-reaching side drawings ghost, because a drawn profile's
# eye and jaw do not land where the front drawing's do on the same surface. Stock baking also drops a 9-texel band around depth edges,
# which removes linework at overhangs such as a beard along the jaw.
PROJECT_WEIGHT, PROJECT_BAKE_ANGLE, PROJECT_EDGE_KERNEL = 200, 82, 1
FIT_SIZE = 256


def square_figure(rgba, fill=0.9):
    """The figure centred on a transparent square. 2.1 Paint squashes its reference to a square, which would distort a tall crop."""
    figure = rgba.crop(rgba.getchannel("A").point(lambda a: 255 if a > 16 else 0).getbbox())
    side = round(max(figure.size) / fill)
    square = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    square.paste(figure, ((side - figure.width) // 2, (side - figure.height) // 2))
    return square


def on_white(rgba, size):
    white = Image.new("RGB", rgba.size, (255, 255, 255))
    white.paste(rgba, mask=rgba.getchannel("A"))
    return white.resize((size, size), Image.BICUBIC)


class MultiReference:
    """Stands in for unet_dual: runs each reference through it and joins their features along the token axis."""

    def __init__(self, unet_dual, vae, extra_images, seed, device):
        self.unet_dual, self.vae, self.extra_images, self.seed, self.device = unet_dual, vae, extra_images, seed, device

    def __call__(self, latents, material_tokens):
        features =[self.unet_dual(latents, material_tokens)]
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        for image in self.extra_images:
            pixels = torch.from_numpy(np.array(image) / 255.0)[None].permute(0, 3, 1, 2)[None].to(device=self.device, dtype=torch.float16)
            features.append(self.unet_dual(paint_pipeline.encode_images(self.vae, pixels, generator)[0], material_tokens))
        return {layer: torch.cat([f[layer] for f in features], dim=1) for layer in features[0]}


def silhouette(normal_map):
    """Mesh pixels of a rendered normal map, whose background is pure white."""
    return np.array(normal_map.convert("RGB")).min(axis=-1) < 255


def fit_view(rgba, target):
    """Scale and shift for rgba's matte that best overlaps the boolean target silhouette: (iou, scale, dx, dy), in target pixels."""
    size = target.shape[0]
    small = Image.fromarray(target).resize((FIT_SIZE, FIT_SIZE), Image.NEAREST)
    mask = np.array(small, dtype=bool)
    matte = rgba.getchannel("A").point(lambda a: 255 if a > 128 else 0)
    box = matte.getbbox()
    figure = matte.crop(box)
    target_rows = np.flatnonzero(mask.any(axis=1))
    target_cols = np.flatnonzero(mask.any(axis=0))
    base = (target_rows[-1] - target_rows[0] + 1) / figure.height
    best = (-1.0, 1.0, 0, 0)
    for factor in np.linspace(0.92, 1.08, 17):
        scale = base * factor
        width, height = max(1, round(figure.width * scale)), max(1, round(figure.height * scale))
        scaled = np.array(figure.resize((width, height), Image.BILINEAR)) > 128
        x_center = (target_cols[0] + target_cols[-1] - width) // 2
        y_center = (target_rows[0] + target_rows[-1] - height) // 2
        for dy in range(-10, 11, 2):
            for dx in range(-24, 25, 2):
                canvas = np.zeros_like(mask)
                x, y = x_center + dx, y_center + dy
                x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + width, FIT_SIZE), min(y + height, FIT_SIZE)
                if x1 <= x0 or y1 <= y0:
                    continue
                canvas[y0:y1, x0:x1] = scaled[y0 - y:y1 - y, x0 - x:x1 - x]
                iou = (canvas & mask).sum() / max((canvas | mask).sum(), 1)
                if iou > best[0]:
                    best = (float(iou), scale, x, y)
    iou, scale, x, y = best
    ratio = size / FIT_SIZE
    return iou, scale * ratio, x * ratio, y * ratio, box


def place_view(rgba, fit, size):
    """The view's figure placed in a transparent square according to fit_view's result."""
    _, scale, x, y, box = fit
    figure = rgba.crop(box)
    figure = figure.resize((max(1, round(figure.width * scale)), max(1, round(figure.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    canvas.paste(figure, (round(x), round(y)))
    return canvas


def facing_weight(normal_map, mesh_mask, elevation, azimuth, inner, outer):
    """1 where the surface is within inner degrees of facing the camera, falling to 0 at outer degrees."""
    normals = np.array(normal_map.convert("RGB"), dtype=np.float32) / 255 * 2 - 1
    facing = normals @ paint_pipeline.get_mv_matrix(elevation, azimuth)[2, :3]
    if facing[mesh_mask].mean() < 0:  # visible surfaces face the camera on average, whichever way the mesh's faces wind
        facing = -facing
    low, high = np.cos(np.radians(outer)), np.cos(np.radians(inner))
    return np.clip((facing - low) / (high - low), 0, 1)


def composite(generated, placed, mesh_mask, inset, feather, region=None):
    """Sheet pixels over the generated view, kept inside both silhouettes so background never reaches the texture."""
    alpha = np.array(placed.getchannel("A")) > 128
    keep = ndimage.binary_erosion(alpha & mesh_mask, iterations=inset)
    weight = ndimage.gaussian_filter(keep.astype(np.float32), feather)[..., None]
    weight = np.clip((weight - 0.5) * 2, 0, 1)  # the feather stays inside the eroded region
    if region is not None:
        weight = weight * region[..., None]
    real, fake = np.array(placed.convert("RGB"), dtype=np.float32), generated.astype(np.float32)
    return (real * weight + fake * (1 - weight)).round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views_dir")
    parser.add_argument("mesh")
    parser.add_argument("output")
    parser.add_argument("--primary", default="front", help="the reference that drives DINO conditioning")
    parser.add_argument("--references", default="front,right,back", help="comma-separated views for reference attention")
    parser.add_argument("--project", action="store_true", help="composite real sheet pixels onto the matching camera views")
    parser.add_argument("--hybrid", action="store_true", help="keep the generated paint and add sheet pixels only where the surface faces the front or back camera")
    parser.add_argument("--facing-angles", default="55,70", help="with --hybrid, surface angles where sheet pixels start to fade and are gone")
    parser.add_argument("--inset", type=int, default=4, help="pixels kept clear of the silhouette edge with --project")
    parser.add_argument("--feather", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--max-faces", type=int, default=100000)
    parser.add_argument("--texture-views", default="8_768")
    parser.add_argument("--texture-size", type=int, default=2048)
    args = parser.parse_args()

    views_dir = Path(args.views_dir)
    views = {path.stem: square_figure(Image.open(path).convert("RGBA")) for path in views_dir.glob("*.png")}
    reference_names = args.references.split(",")
    extras = [name for name in reference_names if name != args.primary]
    report = {"references": reference_names, "project": args.project, "hybrid": args.hybrid, "fits": {}}

    original_diffusion = paint_pipeline.multiview_diffusion

    def multiview_diffusion(unet, unet_dual, vae, dino, reference, normal_maps, position_maps, view_size, steps, guidance, seed, callback, device):
        extra_images = [on_white(views[name], view_size) for name in extras]
        multi = MultiReference(unet_dual, vae, extra_images, seed, device) if extra_images else unet_dual
        albedo, mr = original_diffusion(unet, multi, vae, dino, reference, normal_maps, position_maps, view_size, steps, guidance, seed, callback, device)
        if args.project or args.hybrid:
            for index, name in PROJECTED_CAMERAS.items():
                if name in views:
                    mask = silhouette(normal_maps[index])
                    fit = fit_view(views[name], mask)
                    report["fits"][name] = {"view": name, f"iou_{name}": round(fit[0], 3)}
                    print(f"{name} silhouette fit: {fit[0]:.3f}")
                    region = None
                    if args.hybrid:
                        inner, outer = (float(angle) for angle in args.facing_angles.split(","))
                        region = facing_weight(normal_maps[index], mask, *paint_pipeline.CANDIDATE_VIEWS[index][:2], inner, outer)
                        print(f"{name} sheet pixel coverage: {region[mask].mean():.2f}")
                    albedo[index] = composite(albedo[index], place_view(views[name], fit, view_size), mask, args.inset, args.feather, region)
        return albedo, mr

    paint_pipeline.multiview_diffusion = multiview_diffusion
    if args.hybrid:
        paint_pipeline.UNRELIABLE_KERNEL = PROJECT_EDGE_KERNEL
    if args.project:
        paint_pipeline.BAKE_ANGLE_THRESHOLD = PROJECT_BAKE_ANGLE
        paint_pipeline.UNRELIABLE_KERNEL = PROJECT_EDGE_KERNEL
        for index, name in PROJECTED_CAMERAS.items():
            if name in views:  # (elevation, azimuth, weight)
                elevation, azimuth, _ = paint_pipeline.CANDIDATE_VIEWS[index]
                paint_pipeline.CANDIDATE_VIEWS[index] = (elevation, azimuth, PROJECT_WEIGHT)

    start = time.time()
    # Load the way Wan2GP's own texturer does (postprocessing.mesh3d_models.ModelRuntime): wgp owns the memory manager's setup.
    sys.argv = sys.argv[:1]  # wgp parses the command line when imported
    import wgp

    model, offloadobj = wgp.load_models(MODEL_TYPE, output_type="image", track_as_main=False, extension=True)
    attention = get_default_attention_mode()  # wgp sets these per generation
    offload.shared_state.update({"_attention": attention, "_attention_fallback": attention, "_sage_v_scaling": False})

    source = load_mesh(args.mesh, device=model.device)
    mesh = Mesh(source.vertices.float(), source.faces)
    print(f"mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    textured = call_with_default_device(wgp.get_model_def(MODEL_TYPE), MODEL_TYPE, model.texture_mesh, mesh, views[args.primary], seed=args.seed,
                                        guidance_scale=args.guidance, max_faces=args.max_faces, texture_views=args.texture_views, texture_size=args.texture_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(mesh_to_glb(textured.to("cpu")))
    report["seconds"] = round(time.time() - start, 1)
    report["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    offloadobj.release()


if __name__ == "__main__":
    main()
