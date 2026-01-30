from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory
import xml.etree.ElementTree as ET

import new_toy

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "frontend"
DIST_DIR = WEB_DIR / "dist"
STATE_JSON = ROOT / "ui_state.json"
STATE_SVG = ROOT / "ui_state.svg"
SVG_PATH = ROOT / "convoi.svg"
SVG_BACKUP = ROOT / "convoi_backup.svg"

app = Flask(__name__, static_folder=None)


def ensure_outputs() -> None:
    env = os.environ.copy()
    env["INTERSECT_SNAP"] = str(new_toy.INTERSECT_SNAP)
    env["LINE_EXTEND"] = str(new_toy.LINE_EXTEND)
    cmd = [os.fspath(Path(os.environ.get("PYTHON", "python"))), os.fspath(ROOT / "new_toy.py")]
    subprocess.run(cmd, cwd=ROOT, env=env, check=False)


@app.get("/api/scene")
def api_scene():
    snap = request.args.get("snap", type=float) or new_toy.INTERSECT_SNAP
    for key, env_key in (
        ("pack_padding", "PACK_PADDING"),
        ("pack_margin_x", "PACK_MARGIN_X"),
        ("pack_margin_y", "PACK_MARGIN_Y"),
        ("pack_bleed", "PACK_BLEED"),
        ("pack_grid", "PACK_GRID_STEP"),
        ("pack_angle", "PACK_ANGLE_STEP"),
        ("pack_mode", "PACK_MODE"),
    ):
        val = request.args.get(key)
        if val is not None:
            os.environ[env_key] = str(val)
    data = new_toy.compute_scene(new_toy.SVG_PATH, snap)
    return jsonify(data)


@app.post("/api/render")
def api_render():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    snap = float(payload.get("snap", new_toy.INTERSECT_SNAP))
    env = os.environ.copy()
    env["INTERSECT_SNAP"] = str(snap)
    env["LINE_EXTEND"] = str(new_toy.LINE_EXTEND)
    for key, env_key in (
        ("pack_padding", "PACK_PADDING"),
        ("pack_margin_x", "PACK_MARGIN_X"),
        ("pack_margin_y", "PACK_MARGIN_Y"),
        ("pack_bleed", "PACK_BLEED"),
        ("pack_grid", "PACK_GRID_STEP"),
        ("pack_angle", "PACK_ANGLE_STEP"),
        ("pack_mode", "PACK_MODE"),
    ):
        if key in payload:
            env[env_key] = str(payload[key])
    cmd = [os.fspath(Path(os.environ.get("PYTHON", "python"))), os.fspath(ROOT / "new_toy.py")]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        return jsonify({"ok": False, "error": proc.stderr.strip()}), 500
    return jsonify({"ok": True})


@app.post("/api/state")
def api_state():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    STATE_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # build a lightweight svg for fast restore
    canvas = payload.get("canvas", {})
    w = canvas.get("w", 1000)
    h = canvas.get("h", 1000)
    paths = []
    for region in payload.get("regions", []):
        if not region:
            continue
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in region) + " Z"
        paths.append(f'<path d="{d}" fill="none" stroke="#999" stroke-width="0.5"/>')
    labels = []
    for lbl in payload.get("labels", []):
        labels.append(
            f'<text x="{lbl["x"]}" y="{lbl["y"]}" font-size="6" fill="#000">{lbl["label"]}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">'
        + "".join(paths)
        + "".join(labels)
        + "</svg>"
    )
    STATE_SVG.write_text(svg, encoding="utf-8")
    return jsonify({"ok": True})


@app.post("/api/save_svg")
def api_save_svg():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    nodes = payload.get("nodes", [])
    segs = payload.get("segs", [])
    if not SVG_PATH.exists():
        return jsonify({"ok": False, "error": "convoi.svg not found"}), 404

    if not SVG_BACKUP.exists():
        SVG_BACKUP.write_bytes(SVG_PATH.read_bytes())

    tree = ET.parse(SVG_PATH)
    root = tree.getroot()

    # remove existing line/polyline/polygon (keep image)
    for parent in list(root.iter()):
        for child in list(parent):
            tag = child.tag.rsplit("}", 1)[-1]
            if tag in {"line", "polyline", "polygon"}:
                parent.remove(child)

    ns = {"svg": "http://www.w3.org/2000/svg"}
    g = ET.Element("g", {"id": "INTERACTIVE"})
    for seg in segs:
        try:
            a = nodes[seg[0]]
            b = nodes[seg[1]]
            line = ET.Element(
                "line",
                {
                    "x1": str(a["x"]),
                    "y1": str(a["y"]),
                    "x2": str(b["x"]),
                    "y2": str(b["y"]),
                    "stroke": "#000",
                    "stroke-width": "1",
                    "fill": "none",
                },
            )
            g.append(line)
        except Exception:
            continue
    root.append(g)

    tree.write(SVG_PATH, encoding="utf-8", xml_declaration=True)
    return jsonify({"ok": True})


@app.get("/")
def index():
    if DIST_DIR.exists():
        return send_from_directory(DIST_DIR, "index.html")
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/<path:path>")
def static_proxy(path: str):
    if DIST_DIR.exists() and (DIST_DIR / path).exists():
        return send_from_directory(DIST_DIR, path)
    return send_from_directory(WEB_DIR, path)


@app.get("/out/<path:path>")
def output_files(path: str):
    return send_from_directory(ROOT, path)


if __name__ == "__main__":
    ensure_outputs()
    app.run(host="127.0.0.1", port=5000, debug=True)
