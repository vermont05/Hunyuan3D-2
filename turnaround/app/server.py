#!/usr/bin/env python
"""Local web app: paste a turnaround sheet, generate a textured model, browse the results.

Run with the Hunyuan3D-2 venv:  ~/Hunyuan3D-2/.venv/bin/python ~/Hunyuan3D-2/turnaround/app/server.py
There is no login: anyone who can reach the port can use the GPU and see the results.
"""
import argparse
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageOps

APP = Path(__file__).resolve().parent
ROOT = APP.parent
STATIC = APP / "static"
OUT = ROOT / "out"
WORK = ROOT / "work"
HOME = Path.home()
# Defaults fit the machine this was built on; another machine sets these in its launcher (see em680/start.sh).
HUNYUAN_DIR = Path(os.environ.get("TURNAROUND_HY3DGEN_DIR", HOME / "Hunyuan3D-2"))  # holds the hy3dgen package
HUNYUAN_PYTHON = Path(os.environ.get("TURNAROUND_SHAPE_PYTHON", HUNYUAN_DIR / ".venv/bin/python"))
WAN_DIR = Path(os.environ.get("TURNAROUND_WAN_DIR", HOME / "Wan2GP"))
WAN_PYTHON = Path(os.environ.get("TURNAROUND_PAINT_PYTHON", WAN_DIR / ".venv/bin/python"))

MAX_UPLOAD = 40 * 1024 * 1024
ANALYZE_TIMEOUT = 600
STEP_TIMEOUT = 1800
MIN_FREE_VRAM_MIB = int(os.environ.get("TURNAROUND_MIN_FREE_VRAM_MIB", 8000))  # the shape step peaks at 5.7 GiB
WORK_MAX_AGE = 24 * 3600
VIEW_NAMES = ("front", "left", "back", "right")
LABELS = VIEW_NAMES + ("skip",)
VIEW_ORDER = ("front", "right", "back", "left")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Texture styles, in the order they are painted and listed: (output file, paint_mv.py flags).
STYLES = {"hybrid": ("textured_hybrid.glb", ["--hybrid"]), "inferred": ("textured_inferred.glb", []), "projection": ("textured.glb", ["--project"])}
VERSION_LABELS = {
    "textured_hybrid.glb": "Hybrid", "textured_inferred.glb": "Inferred", "textured.glb": "Projection", "textured_project.glb": "Projection",
    "textured_before_fix.glb": "Projection, old blend", "textured_stock.glb": "Front reference only",
    "shape.glb": "Shape", "shape_4view.glb": "Shape, 4 views", "shape_3view.glb": "Shape, 3 views",
}
ENV = {**os.environ, "U2NET_HOME": os.environ.get("U2NET_HOME", str(WAN_DIR / "ckpts/rembg")), "PYTHONPATH": str(HUNYUAN_DIR)}

jobs = queue.Queue()
meta_lock = threading.Lock()
analysis_lock = threading.Lock()


class UserError(Exception):
    """A problem to show the user as is."""


def read_meta(run_dir):
    try:
        return json.loads((run_dir / "meta.json").read_text())
    except (OSError, ValueError):
        return {}


def update_meta(run_dir, **fields):
    with meta_lock:
        meta = read_meta(run_dir)
        meta.update(fields)
        temp = run_dir / "meta.json.tmp"
        temp.write_text(json.dumps(meta, indent=2))
        temp.replace(run_dir / "meta.json")


def last_error_line(text):
    lines = [line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip()]
    for line in reversed(lines):
        if "Error" in line or "error" in line:
            return line[:300]
    return lines[-1][:300] if lines else "The step failed without any output."


def run_step(command, cwd, log_path):
    with open(log_path, "a") as log:
        log.write(f"\n$ {' '.join(map(str, command))}\n")
        log.flush()
        try:
            code = subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, stdout=log, stderr=subprocess.STDOUT, timeout=STEP_TIMEOUT).returncode
        except subprocess.TimeoutExpired:
            raise UserError(f"The step ran past the {STEP_TIMEOUT // 60} minute limit and was stopped.")
    if code != 0:
        raise UserError(last_error_line(Path(log_path).read_text(errors="replace")[-8000:]))


def free_vram_mib():
    if shutil.which("nvidia-smi"):
        output = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
        return int(output.split()[0])
    # AMD: the amdgpu driver's counters for the card with the most VRAM, which is the one the launcher selects.
    cards = []
    for device in Path("/sys/class/drm").glob("card*/device"):
        try:
            total, used = (int((device / f"mem_info_vram_{name}").read_text()) for name in ("total", "used"))
        except (OSError, ValueError):
            continue
        cards.append((total, total - used))
    if not cards:
        raise UserError("Couldn't find a GPU to check its free memory.")
    return max(cards)[1] // 2**20


def process(run_dir):
    log = run_dir / "job.log"
    views = run_dir / "views"
    seconds = {}
    try:
        free = free_vram_mib()
        if free < MIN_FREE_VRAM_MIB:
            raise UserError(f"Only {free / 1024:.1f} GiB of VRAM is free and this needs about {MIN_FREE_VRAM_MIB / 1024:.0f} GiB. "
                            "Stop whatever is using the GPU, such as basalt, then generate again.")
        update_meta(run_dir, status="running", stage="shape")
        start = time.time()
        run_step([HUNYUAN_PYTHON, ROOT / "shape_mv.py", views, run_dir / "shape.glb"], ROOT, log)
        seconds["shape"] = round(time.time() - start, 1)
        meta = read_meta(run_dir)
        references = ",".join(meta["references"])
        for style in meta.get("styles", ["projection"]):
            output, flags = STYLES[style]
            update_meta(run_dir, stage=f"paint_{style}", seconds=seconds)
            start = time.time()
            run_step([WAN_PYTHON, ROOT / "paint_mv.py", views, run_dir / "shape.glb", run_dir / output, *flags, "--references", references], WAN_DIR, log)
            seconds[f"paint_{style}"] = round(time.time() - start, 1)
        update_meta(run_dir, status="done", stage=None, seconds=seconds)
    except UserError as error:
        update_meta(run_dir, status="failed", stage=None, error=str(error), seconds=seconds)
    except Exception as error:
        update_meta(run_dir, status="failed", stage=None, error=f"{type(error).__name__}: {error}"[:300], seconds=seconds)


def worker():
    while True:
        process(jobs.get())


def analyze(data):
    if not data:
        raise UserError("No image was received.")
    upload_id = uuid.uuid4().hex
    folder = WORK / upload_id
    folder.mkdir(parents=True)
    upload = folder / "upload"
    upload.write_bytes(data)
    with analysis_lock:
        try:
            result = subprocess.run([str(HUNYUAN_PYTHON), str(ROOT / "analyze_sheet.py"), str(upload), str(folder)], cwd=ROOT, env=ENV,
                                    capture_output=True, text=True, timeout=ANALYZE_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise UserError("Finding the views took too long and was stopped.")
    upload.unlink(missing_ok=True)
    if result.returncode != 0:
        shutil.rmtree(folder, ignore_errors=True)
        raise UserError(last_error_line(result.stderr))
    analysis = json.loads((folder / "analysis.json").read_text())
    return {"id": upload_id, "sheet": f"work/{upload_id}/sheet.png",
            "figures": [{"file": f["file"], "url": f"work/{upload_id}/{f['file']}", "label": f["label"]} for f in analysis["figures"]]}


def unique_run_name(requested):
    base = re.sub(r"[^a-z0-9]+", "-", (requested or "").lower()).strip("-")[:48] or time.strftime("sheet-%Y%m%d-%H%M%S")
    name, suffix = base, 2
    while (OUT / name).exists():
        name, suffix = f"{base}-{suffix}", suffix + 1
    return name


def generate(body):
    upload_id = str(body.get("id", ""))
    folder = WORK / upload_id
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id) or not folder.is_dir():
        raise UserError("That upload has expired. Paste the image again.")
    labels = body.get("labels")
    if not isinstance(labels, dict):
        raise UserError("The view labels are missing.")
    analysis = json.loads((folder / "analysis.json").read_text())
    files = {figure["file"] for figure in analysis["figures"]}
    if set(labels) != files or any(label not in LABELS for label in labels.values()):
        raise UserError("The view labels don't match this image. Paste the image again.")
    chosen = {label: file for file, label in labels.items() if label != "skip"}
    if len(chosen) != sum(label != "skip" for label in labels.values()):
        raise UserError("Each view can be used only once. Set the extra one to Skip.")
    if "front" not in chosen:
        raise UserError("Mark one figure as Front. The texture needs it.")
    if len(chosen) < 2:
        raise UserError("Mark at least two views.")
    styles = body.get("styles", ["hybrid"])
    if not isinstance(styles, list) or not styles or any(style not in STYLES for style in styles):
        raise UserError("Choose at least one texture style.")
    styles = [style for style in STYLES if style in styles]

    run_dir = OUT / unique_run_name(body.get("name"))
    views = run_dir / "views"
    views.mkdir(parents=True)
    for label, file in chosen.items():
        shutil.copyfile(folder / file, views / f"{label}.png")
    mirrored = None
    sides = {"left": "right", "right": "left"}
    present = [side for side in sides if side in chosen]
    if len(present) == 1:  # the fourth view gives the shape step a better read of the far side
        mirrored = sides[present[0]]
        ImageOps.mirror(Image.open(views / f"{present[0]}.png")).save(views / f"{mirrored}.png")
    shutil.copyfile(folder / "sheet.png", run_dir / "sheet.png")
    references = [view for view in VIEW_ORDER if view in chosen]
    update_meta(run_dir, status="queued", stage=None, created=time.time(), references=references, mirrored=mirrored, styles=styles)
    shutil.rmtree(folder, ignore_errors=True)
    jobs.put(run_dir)
    return {"run": run_dir.name}


def describe_run(run_dir):
    meta = read_meta(run_dir)
    versions = []
    glbs = {path.name: path for path in run_dir.glob("*.glb")}
    for name in [*VERSION_LABELS, *sorted(set(glbs) - set(VERSION_LABELS))]:
        if name in glbs:
            versions.append({"file": name, "label": VERSION_LABELS.get(name, name), "url": f"runs/{run_dir.name}/{name}",
                             "bytes": glbs[name].stat().st_size, "path": str(glbs[name]).replace(str(HOME), "~", 1)})
    scores = {}
    for name in ("textured_hybrid.json", "textured.json", "textured_project.json"):
        try:
            for fit in json.loads((run_dir / name).read_text())["fits"].values():
                scores[fit["view"]] = fit[f"iou_{fit['view']}"]
            break
        except (OSError, ValueError, KeyError):
            continue
    views = [{"name": view, "url": f"runs/{run_dir.name}/views/{view}.png", "score": scores.get(view), "mirrored": meta.get("mirrored") == view}
             for view in VIEW_ORDER if (run_dir / "views" / f"{view}.png").exists()]
    created = meta.get("created") or run_dir.stat().st_mtime
    return {"name": run_dir.name, "status": meta.get("status", "done" if versions else "failed"), "stage": meta.get("stage"), "error": meta.get("error"),
            "created": created, "seconds": meta.get("seconds", {}), "versions": versions, "views": views,
            "sheet": f"runs/{run_dir.name}/sheet.png" if (run_dir / "sheet.png").exists() else None}


def list_runs():
    runs = [describe_run(path) for path in OUT.iterdir() if path.is_dir() and SAFE_NAME.match(path.name)] if OUT.is_dir() else []
    return sorted(runs, key=lambda run: run["created"], reverse=True)


def prepare():
    OUT.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    for path in OUT.iterdir():  # nothing survives a restart: the queue lives in this process
        if path.is_dir() and read_meta(path).get("status") in ("queued", "running"):
            update_meta(path, status="failed", stage=None, error="The server restarted before this finished. Generate it again.")
    for path in WORK.iterdir():
        if path.is_dir() and time.time() - path.stat().st_mtime > WORK_MAX_AGE:
            shutil.rmtree(path, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def send_json(self, value, status=200):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path):
        if not path.is_file():
            return self.send_json({"error": "Not found."}, 404)
        kind = "model/gltf-binary" if path.suffix == ".glb" else mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(path, "rb") as source:
            shutil.copyfileobj(source, self.wfile)

    def safe_file(self, base, parts):
        """base/parts, or None unless every part is a plain name and the file type is one the app serves."""
        if not parts or not all(SAFE_NAME.match(part) for part in parts):
            return None
        return base.joinpath(*parts) if Path(parts[-1]).suffix in (".png", ".glb", ".js", ".html") else None

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        parts = [part for part in path.split("/") if part]
        try:
            if not parts:
                return self.send_file(STATIC / "index.html")
            if parts == ["api", "runs"]:
                return self.send_json({"runs": list_runs()})
            roots = {"static": STATIC, "runs": OUT, "work": WORK}
            target = self.safe_file(roots[parts[0]], parts[1:]) if parts[0] in roots else None
            if target is None:
                return self.send_json({"error": "Not found."}, 404)
            self.send_file(target)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        try:
            if length > MAX_UPLOAD:
                self.close_connection = True
                return self.send_json({"error": f"That image is larger than {MAX_UPLOAD // 2**20} MB."}, 413)
            data = self.rfile.read(length)
            if self.path == "/api/analyze":
                return self.send_json(analyze(data))
            if self.path == "/api/generate":
                return self.send_json(generate(json.loads(data)))
            self.send_json({"error": "Not found."}, 404)
        except UserError as error:
            self.send_json({"error": str(error)}, 422)
        except ValueError:
            self.send_json({"error": "That request could not be read."}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    prepare()
    threading.Thread(target=worker, daemon=True).start()
    print(f"serving on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
