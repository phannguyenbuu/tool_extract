from __future__ import annotations

import json
import os
import subprocess
import math
import re
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

import new_toy
from scripts import packing, zones, config

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "frontend"
DIST_DIR = WEB_DIR / "dist"
STATE_JSON = ROOT / "ui_state.json"
STATE_SVG = ROOT / "ui_state.svg"
PACKED_LABELS_JSON = ROOT / "packed_labels.json"
ZONE_LABELS_JSON = ROOT / "zone_labels.json"
SCENE_JSON = ROOT / "scene_cache.json"
SOURCE_ZONE_CLICK_JSON = ROOT / "soure_zone_click.json"
PACKED_ZONE_SCENE_SVG = ROOT / "packed_zone_scene.svg"
PACKED_ZONE_SCENE_SVG_PAGE2 = ROOT / "packed_zone_scene_page2.svg"
SVG_PATH = ROOT / "convoi.svg"
SVG_BACKUP = ROOT / "convoi_backup.svg"
EXPORT_DIR = ROOT / "export"

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
        ("draw_scale", "DRAW_SCALE"),
        ("pack_grid", "PACK_GRID_STEP"),
        ("pack_angle", "PACK_ANGLE_STEP"),
        ("pack_mode", "PACK_MODE"),
    ):
        val = request.args.get(key)
        if val is not None:
            os.environ[env_key] = str(val)
    data = new_toy.compute_scene(new_toy.SVG_PATH, snap, render_packed_png=False)
    SCENE_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    zone_labels = data.get("zone_labels")
    if isinstance(zone_labels, dict):
        ZONE_LABELS_JSON.write_text(
            json.dumps(zone_labels, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    # keep packed.svg in sync for Konva vector preview
    try:
        canvas = data.get("canvas") or {}
        w = int(canvas.get("w", 0))
        h = int(canvas.get("h", 0))
        if w > 0 and h > 0:
            new_toy.write_pack_svg(
                data.get("regions", []),
                data.get("zone_id", []),
                data.get("zone_order", []),
                [],
                data.get("placements", []),
                (w, h),
                data.get("colors_bgr", []),
                data.get("rot_info", []),
            )
    except Exception:
        pass
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
        ("draw_scale", "DRAW_SCALE"),
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


@app.get("/api/state")
def api_get_state():
    if STATE_JSON.exists():
        try:
            data = json.loads(STATE_JSON.read_text(encoding="utf-8"))
            return jsonify(data)
        except Exception:
            return jsonify({})
    return jsonify({})


@app.get("/api/packed_labels")
def api_packed_labels():
    if PACKED_LABELS_JSON.exists():
        try:
            data = json.loads(PACKED_LABELS_JSON.read_text(encoding="utf-8"))
            return jsonify(data)
        except Exception:
            return jsonify({})
    return jsonify({})


@app.post("/api/packed_labels")
def api_save_packed_labels():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    data: Dict[str, Any] = {}
    if PACKED_LABELS_JSON.exists():
        try:
            data = json.loads(PACKED_LABELS_JSON.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    for key, val in payload.items():
        data[str(key)] = val
    PACKED_LABELS_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"ok": True})


def _color_to_bgr(val: Any) -> tuple[int, int, int]:
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("#"):
            h = s[1:]
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            if len(h) == 6:
                try:
                    r = int(h[0:2], 16)
                    g = int(h[2:4], 16)
                    b = int(h[4:6], 16)
                    return (b, g, r)
                except Exception:
                    return (200, 200, 200)
        if s.lower().startswith("rgb"):
            nums = re.findall(r"[-+]?\d+", s)
            if len(nums) >= 3:
                try:
                    r = int(nums[0])
                    g = int(nums[1])
                    b = int(nums[2])
                    return (b, g, r)
                except Exception:
                    return (200, 200, 200)
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        try:
            r = int(val[0])
            g = int(val[1])
            b = int(val[2])
            return (b, g, r)
        except Exception:
            return (200, 200, 200)
    return (200, 200, 200)


@app.post("/api/pack_from_scene")
def api_pack_from_scene():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    canvas = payload.get("canvas") or {}
    w = int(canvas.get("w", 0) or 0)
    h = int(canvas.get("h", 0) or 0)
    if w <= 0 or h <= 0:
        return jsonify({"ok": False, "error": "canvas missing"}), 400
    regions = payload.get("regions") or []
    zone_id = payload.get("zone_id") or []
    if not regions or not zone_id:
        return jsonify({"ok": False, "error": "regions/zone_id missing"}), 400
    config._apply_pack_env()
    polys = []
    for poly in regions:
        pts = []
        for p in poly or []:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                pts.append((float(p[0]), float(p[1])))
            except Exception:
                continue
        if pts:
            polys.append(pts)
        else:
            polys.append([])
    zone_polys, zone_order, _zone_poly_debug = zones.build_zone_polys(polys, zone_id)
    zone_pack_polys = packing._build_zone_pack_polys(
        zone_polys, float(config.PACK_BLEED), bevel_angle=60.0
    )
    zone_pack_centers = []
    for p in zone_pack_polys:
        if p:
            pg = packing.Polygon(p)
            if pg.is_empty:
                zone_pack_centers.append((0.0, 0.0))
            else:
                c = pg.centroid
                zone_pack_centers.append((float(c.x), float(c.y)))
        else:
            zone_pack_centers.append((0.0, 0.0))
    placements, _order, rot_info = packing.pack_regions(
        zone_pack_polys,
        (w, h),
        allow_rotate=True,
        angle_step=5.0,
        grid_step=config.PACK_GRID_STEP,
        fixed_angles=None,
        fixed_centers=zone_pack_centers,
        max_bins=2,
        try_heuristics=True,
        two_pass=True,
        preferred_indices=None,
        use_gap_only=False,
    )
    placement_bin = [int(info.get("bin", -1)) for info in rot_info]
    placement_bin_by_zid = {
        zid: placement_bin[idx]
        for idx, zid in enumerate(zone_order)
        if idx < len(placement_bin)
    }
    zone_shift: Dict[int, tuple[float, float]] = {}
    zone_rot: Dict[int, float] = {}
    zone_center: Dict[int, tuple[float, float]] = {}
    for idx, zid in enumerate(zone_order):
        if idx >= len(placements):
            continue
        dx, dy, _, _, _ = placements[idx]
        zone_shift[zid] = (float(dx), float(dy))
        info = rot_info[idx] if idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
        zone_rot[zid] = float(info.get("angle", 0.0))
        zone_center[zid] = (float(info.get("cx", 0.0)), float(info.get("cy", 0.0)))
    colors_raw = payload.get("region_colors") or []
    colors = []
    for i in range(len(polys)):
        c = colors_raw[i] if i < len(colors_raw) else None
        colors.append(_color_to_bgr(c))
    packing.write_pack_svg(
        polys,
        zone_id,
        zone_order,
        zone_polys,
        placements,
        (w, h),
        colors,
        rot_info,
        placement_bin=placement_bin,
        placement_bin_by_zid=placement_bin_by_zid,
        page_idx=0,
        out_path=PACKED_ZONE_SCENE_SVG,
    )
    packing.write_pack_svg(
        polys,
        zone_id,
        zone_order,
        zone_polys,
        placements,
        (w, h),
        colors,
        rot_info,
        placement_bin=placement_bin,
        placement_bin_by_zid=placement_bin_by_zid,
        page_idx=1,
        out_path=PACKED_ZONE_SCENE_SVG_PAGE2,
    )
    return jsonify(
        {
            "ok": True,
            "packed_svg": PACKED_ZONE_SCENE_SVG.read_text(encoding="utf-8"),
            "packed_svg_page2": PACKED_ZONE_SCENE_SVG_PAGE2.read_text(encoding="utf-8"),
            "zone_shift": zone_shift,
            "zone_rot": zone_rot,
            "zone_center": zone_center,
            "placement_bin": placement_bin_by_zid,
        }
    )


@app.get("/api/source_zone_click")
def api_get_source_zone_click():
    if SOURCE_ZONE_CLICK_JSON.exists():
        try:
            data = json.loads(SOURCE_ZONE_CLICK_JSON.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return jsonify({"clicks": data})
            if isinstance(data, dict):
                return jsonify({"clicks": data.get("clicks", [])})
        except Exception:
            return jsonify({"clicks": []})
    return jsonify({"clicks": []})


@app.post("/api/source_zone_click")
def api_save_source_zone_click():
    payload: Any = request.get_json(silent=True) or []
    clicks = payload if isinstance(payload, list) else payload.get("clicks", [])
    cleaned = []
    for item in clicks or []:
        if not isinstance(item, dict):
            continue
        x = item.get("x")
        y = item.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if math.isfinite(x) and math.isfinite(y):
                cleaned.append({"x": float(x), "y": float(y)})
    SOURCE_ZONE_CLICK_JSON.write_text(
        json.dumps({"clicks": cleaned}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return jsonify({"ok": True, "count": len(cleaned)})


@app.post("/api/export")
def api_export():
    try:
        print("[export] 0% start")
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        print("[export] 10% use existing outputs")

        target_w_mm = 260.0
        target_h_mm = 190.0
        packed_labels_fallback: Dict[str, Any] = {}
        print("[export] 11% packed labels=ignored (use zone_labels)")

        canvas_w = None
        canvas_h = None
        cached_scene: Dict[str, Any] = {}
        if SCENE_JSON.exists():
            try:
                cached_scene = json.loads(SCENE_JSON.read_text(encoding="utf-8"))
            except Exception:
                cached_scene = {}
        print(
            f"[export] 12% scene cache keys={len(cached_scene)} canvas={bool(cached_scene.get('canvas'))}"
        )
        if cached_scene.get("canvas"):
            try:
                canvas_w = float(cached_scene["canvas"].get("w", 0))
                canvas_h = float(cached_scene["canvas"].get("h", 0))
            except Exception:
                canvas_w = None
                canvas_h = None
            # fallback labels from scene cache
            if cached_scene.get("zone_labels"):
                packed_labels_fallback = cached_scene.get("zone_labels", {})
        if canvas_w and canvas_h:
            print(f"[export] 13% canvas={int(canvas_w)}x{int(canvas_h)}")
        else:
            print("[export] 13% canvas=missing")

        prefix = new_toy.config.SVG_PATH.stem
        print("[export] 90% write svgs")
        prefix = new_toy.config.SVG_PATH.stem
        zone_outline_svg = ROOT / "zone_outline.svg"
        packed_svg = ROOT / "packed.svg"

        if canvas_w and canvas_h and cached_scene.get("zone_boundaries"):
            def rotate_pt(pt, angle_deg, cx, cy):
                if not angle_deg:
                    return pt
                ang = (angle_deg * math.pi) / 180.0
                c = math.cos(ang)
                s = math.sin(ang)
                x = pt[0] - cx
                y = pt[1] - cy
                return [cx + x * c - y * s, cy + x * s + y * c]

            def transform_path(pts, shift, rot, center):
                dx = shift[0] if shift else 0
                dy = shift[1] if shift else 0
                ang = rot if rot else 0
                cx = center[0] if center else 0
                cy = center[1] if center else 0
                out = []
                for p in pts:
                    rp = rotate_pt(p, ang, cx, cy)
                    out.append([rp[0] + dx, rp[1] + dy])
                return out

            parts = [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{target_w_mm}mm" height="{target_h_mm}mm" viewBox="0 0 {int(canvas_w)} {int(canvas_h)}">'
            ]
            parts.append(
                f'<rect x="0" y="0" width="{int(canvas_w)}" height="{int(canvas_h)}" '
                f'fill="none" stroke="#ffffff" stroke-width="2"/>'
            )
            for zid, paths in cached_scene.get("zone_boundaries", {}).items():
                shift = cached_scene.get("zone_shift", {}).get(str(zid))
                if shift is None:
                    shift = cached_scene.get("zone_shift", {}).get(int(zid)) if str(zid).isdigit() else None
                rot = cached_scene.get("zone_rot", {}).get(str(zid))
                if rot is None:
                    rot = cached_scene.get("zone_rot", {}).get(int(zid)) if str(zid).isdigit() else 0
                center = cached_scene.get("zone_center", {}).get(str(zid))
                if center is None:
                    center = (
                        cached_scene.get("zone_center", {}).get(int(zid))
                        if str(zid).isdigit()
                        else [0, 0]
                    )
                for poly in paths or []:
                    tpts = transform_path(poly, shift, rot or 0, center or [0, 0])
                    if not tpts:
                        continue
                    d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in tpts) + " Z"
                    parts.append(f'<path d="{d}" fill="none" stroke="#ffffff" stroke-width="1"/>')
            packed_label_size = float(new_toy.config.PACK_LABEL_SCALE) * 20.0 * 0.25
            for lbl in packed_labels.values():
                try:
                    x = float(lbl.get("x", 0))
                    y = float(lbl.get("y", 0))
                    text = str(lbl.get("label", ""))
                except Exception:
                    continue
                parts.append(
                    f'<text x="{x}" y="{y}" fill="#ffffff" stroke="rgba(0,0,0,0.5)" '
                    f'stroke-width="1" font-size="{packed_label_size}" text-anchor="middle" '
                    f'dominant-baseline="middle">{text}</text>'
                )
            parts.append("</svg>")
            (EXPORT_DIR / f"{prefix}_packed_260x190.svg").write_text("".join(parts), encoding="utf-8")

            # Packed (Konva) stroke-only (no color) + labels
            parts = [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{target_w_mm}mm" height="{target_h_mm}mm" viewBox="0 0 {int(canvas_w)} {int(canvas_h)}">'
            ]
            parts.append(
                f'<rect x="0" y="0" width="{int(canvas_w)}" height="{int(canvas_h)}" '
                f'fill="none" stroke="#ffffff" stroke-width="2"/>'
            )
            for zid, paths in cached_scene.get("zone_boundaries", {}).items():
                shift = cached_scene.get("zone_shift", {}).get(str(zid))
                if shift is None:
                    shift = cached_scene.get("zone_shift", {}).get(int(zid)) if str(zid).isdigit() else None
                rot = cached_scene.get("zone_rot", {}).get(str(zid))
                if rot is None:
                    rot = cached_scene.get("zone_rot", {}).get(int(zid)) if str(zid).isdigit() else 0
                center = cached_scene.get("zone_center", {}).get(str(zid))
                if center is None:
                    center = (
                        cached_scene.get("zone_center", {}).get(int(zid))
                        if str(zid).isdigit()
                        else [0, 0]
                    )
                for poly in paths or []:
                    tpts = transform_path(poly, shift, rot or 0, center or [0, 0])
                    if not tpts:
                        continue
                    d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in tpts) + " Z"
                    parts.append(f'<path d="{d}" fill="none" stroke="#ffffff" stroke-width="1"/>')
            packed_label_size = float(new_toy.config.PACK_LABEL_SCALE) * 20.0 * 0.25
            label_source: Dict[str, Any] = {}
            if packed_labels_fallback:
                # Build packed labels by transforming zone_labels into packed space.
                for zid, lbl in packed_labels_fallback.items():
                    try:
                        x = float(lbl.get("x", 0))
                        y = float(lbl.get("y", 0))
                        text = str(lbl.get("label", ""))
                    except Exception:
                        continue
                    shift = cached_scene.get("zone_shift", {}).get(str(zid))
                    if shift is None:
                        shift = cached_scene.get("zone_shift", {}).get(int(zid)) if str(zid).isdigit() else None
                    rot = cached_scene.get("zone_rot", {}).get(str(zid))
                    if rot is None:
                        rot = cached_scene.get("zone_rot", {}).get(int(zid)) if str(zid).isdigit() else 0
                    center = cached_scene.get("zone_center", {}).get(str(zid))
                    if center is None:
                        center = (
                            cached_scene.get("zone_center", {}).get(int(zid))
                            if str(zid).isdigit()
                            else [0, 0]
                        )
                    tx, ty = transform_path([[x, y]], shift, rot or 0, center or [0, 0])[0]
                    label_source[str(zid)] = {"x": tx, "y": ty, "label": text}
            for lbl in label_source.values():
                try:
                    x = float(lbl.get("x", 0))
                    y = float(lbl.get("y", 0))
                    text = str(lbl.get("label", ""))
                except Exception:
                    continue
                parts.append(
                    f'<text x="{x}" y="{y}" fill="#ffffff" stroke="rgba(0,0,0,0.5)" '
                    f'stroke-width="1" font-size="{packed_label_size}" text-anchor="middle" '
                    f'dominant-baseline="middle">{text}</text>'
                )
            parts.append("</svg>")
            (EXPORT_DIR / f"{prefix}_packed_stroke_260x190.svg").write_text("".join(parts), encoding="utf-8")

        # Packed (Konva) color-only (no stroke) + labels, from packed.svg fill+bleed layers.
        if canvas_w and canvas_h and packed_svg.exists():
            try:
                tree = ET.parse(packed_svg)
                root = tree.getroot()
                ns = {"svg": "http://www.w3.org/2000/svg"}
                fill_group = root.find(".//svg:g[@id='fill']", ns)
                bleed_group = root.find(".//svg:g[@id='bleed']", ns)
                parts = [
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{target_w_mm}mm" height="{target_h_mm}mm" viewBox="0 0 {int(canvas_w)} {int(canvas_h)}">'
                ]
                if fill_group is not None:
                    for p in fill_group.findall(".//svg:path", ns):
                        d = p.attrib.get("d")
                        fill = p.attrib.get("fill", "#ffffff")
                        if not d:
                            continue
                        parts.append(f'<path d="{d}" fill="{fill}" stroke="none"/>')
                if bleed_group is not None:
                    for p in bleed_group.findall(".//svg:path", ns):
                        d = p.attrib.get("d")
                        fill = p.attrib.get("fill", "#ffffff")
                        if not d:
                            continue
                        parts.append(f'<path d="{d}" fill="{fill}" stroke="none"/>')
                packed_label_size = float(new_toy.config.PACK_LABEL_SCALE) * 20.0 * 0.25
                label_source = {}
                if packed_labels_fallback:
                    for zid, lbl in packed_labels_fallback.items():
                        try:
                            x = float(lbl.get("x", 0))
                            y = float(lbl.get("y", 0))
                            text = str(lbl.get("label", ""))
                        except Exception:
                            continue
                        shift = cached_scene.get("zone_shift", {}).get(str(zid))
                        if shift is None:
                            shift = cached_scene.get("zone_shift", {}).get(int(zid)) if str(zid).isdigit() else None
                        rot = cached_scene.get("zone_rot", {}).get(str(zid))
                        if rot is None:
                            rot = cached_scene.get("zone_rot", {}).get(int(zid)) if str(zid).isdigit() else 0
                        center = cached_scene.get("zone_center", {}).get(str(zid))
                        if center is None:
                            center = (
                                cached_scene.get("zone_center", {}).get(int(zid))
                                if str(zid).isdigit()
                                else [0, 0]
                            )
                        tx, ty = transform_path([[x, y]], shift, rot or 0, center or [0, 0])[0]
                        label_source[str(zid)] = {"x": tx, "y": ty, "label": text}
                for lbl in label_source.values():
                    try:
                        x = float(lbl.get("x", 0))
                        y = float(lbl.get("y", 0))
                        text = str(lbl.get("label", ""))
                    except Exception:
                        continue
                    parts.append(
                        f'<text x="{x}" y="{y}" fill="#ffffff" stroke="rgba(0,0,0,0.5)" '
                        f'stroke-width="1" font-size="{packed_label_size}" text-anchor="middle" '
                        f'dominant-baseline="middle">{text}</text>'
                    )
                parts.append("</svg>")
                (EXPORT_DIR / f"{prefix}_packed_color_260x190.svg").write_text(
                    "".join(parts), encoding="utf-8"
                )
            except Exception:
                pass

        if zone_outline_svg.exists() and canvas_w and canvas_h:
            tree = ET.parse(zone_outline_svg)
            root = tree.getroot()
            parts = [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{target_w_mm}mm" height="{target_h_mm}mm" viewBox="0 0 {int(canvas_w)} {int(canvas_h)}">'
            ]
            for p in root.findall(".//{http://www.w3.org/2000/svg}path"):
                d = p.attrib.get("d")
                if not d:
                    continue
                parts.append(f'<path d="{d}" fill="none" stroke="#ffffff" stroke-width="1"/>')
            labels = {}
            if ZONE_LABELS_JSON.exists():
                try:
                    labels = json.loads(ZONE_LABELS_JSON.read_text(encoding="utf-8"))
                except Exception:
                    labels = {}
            font_size = str(float(new_toy.config.PACK_LABEL_SCALE) * 20.0 * 0.2)
            for lbl in labels.values():
                try:
                    x = float(lbl.get("x", 0))
                    y = float(lbl.get("y", 0))
                    text = str(lbl.get("label", ""))
                except Exception:
                    continue
                parts.append(
                    f'<text x="{x}" y="{y}" fill="#ffffff" font-size="{font_size}" '
                    f'text-anchor="middle" dominant-baseline="middle">{text}</text>'
                )
            parts.append("</svg>")
            (EXPORT_DIR / f"{prefix}_zone_260x190.svg").write_text("".join(parts), encoding="utf-8")
        print("[export] 100% done")
        return jsonify({"ok": True})
    except Exception as exc:
        print(f"[export] error: {exc}")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/save_konva_svg")
def api_save_konva_svg():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    svg = payload.get("svg", "")
    if not name or not svg:
        return jsonify({"ok": False, "error": "missing name/svg"}), 400
    safe_name = os.path.basename(name)
    if not safe_name.lower().endswith(".svg"):
        safe_name = f"{safe_name}.svg"
    prefix = f"{SVG_PATH.stem}_"
    if not safe_name.startswith(prefix):
        safe_name = f"{prefix}{safe_name}"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPORT_DIR / safe_name
    try:
        tree = ET.ElementTree(ET.fromstring(svg))
        root = tree.getroot()
        ns = {"svg": "http://www.w3.org/2000/svg"}
        for rect in root.findall(".//svg:rect", ns):
            rect.set("fill", "none")
            rect.set("stroke", "#000000")
            rect.set("stroke-width", "1")
        for text in root.findall(".//svg:text", ns):
            text.set("fill", "#000000")
            text.set("stroke", "none")
            text.set("stroke-width", "0")
            text.set("font-weight", "100")
            text.set("font-family", "Arial, sans-serif")
            text.set("text-anchor", "middle")
            text.set("dominant-baseline", "middle")
            text.set("alignment-baseline", "middle")
            size = text.get("font-size")
            if size:
                try:
                    text.set("font-size", str(float(size) * 0.5))
                except Exception:
                    pass
        for elem in root.iter():
            if elem.tag.endswith("text"):
                continue
            if "stroke" in elem.attrib:
                elem.set("stroke", "#000000")
                elem.set("stroke-width", "1")
        svg = ET.tostring(root, encoding="unicode")
    except Exception:
        pass
    out_path.write_text(svg, encoding="utf-8")
    return jsonify({"ok": True, "path": str(out_path)})


@app.post("/api/save_html")
def api_save_html():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    html = payload.get("html", "")
    if not name or not html:
        return jsonify({"ok": False, "error": "missing name/html"}), 400
    safe_name = os.path.basename(name)
    if not safe_name.lower().endswith(".html"):
        safe_name = f"{safe_name}.html"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPORT_DIR / safe_name
    out_path.write_text(html, encoding="utf-8")
    return jsonify({"ok": True, "path": str(out_path), "name": out_path.name})


@app.post("/api/export_pdf")
def api_export_pdf():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    pages = payload.get("pages") or []
    if not isinstance(pages, list) or not pages:
        return jsonify({"ok": False, "error": "no pages"}), 400
    font_name = str(payload.get("fontName") or "Arial")
    font_size = payload.get("fontSize")
    try:
        font_size = float(font_size) / 2.0
    except Exception:
        font_size = 6.0
    try:
        from reportlab.pdfgen import canvas as pdf_canvas
        from reportlab.graphics import renderPDF
        from svglib.svglib import svg2rlg
    except Exception:
        return jsonify(
            {
                "ok": False,
                "error": "Missing reportlab/svglib. Install: pip install reportlab svglib",
            }
        ), 500

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPORT_DIR / f"{SVG_PATH.stem}_konva.pdf"
    c = pdf_canvas.Canvas(str(out_path))
    tmp_paths: list[Path] = []
    try:
        for idx, page in enumerate(pages):
            svg = page.get("svg", "")
            if not svg:
                continue
            try:
                root = ET.fromstring(svg)
                ns = {"svg": "http://www.w3.org/2000/svg"}
                for rect in root.findall(".//svg:rect", ns):
                    rect.set("fill", "none")
                    rect.set("stroke", "#000000")
                    rect.set("stroke-width", "1")
                for elem in root.iter():
                    if "stroke" in elem.attrib and not elem.tag.endswith("text"):
                        elem.set("stroke", "#000000")
                        elem.set("stroke-width", "1")
                for text in root.findall(".//svg:text", ns):
                    text.set("fill", "#000000")
                    text.set("stroke", "none")
                    text.set("stroke-width", "0")
                    text.set("font-family", font_name)
                    text.set("font-weight", "100")
                    text.set("font-size", str(font_size))
                    text.set("text-anchor", "middle")
                    text.set("dominant-baseline", "middle")
                    text.set("alignment-baseline", "middle")
                    for key in ("x", "y"):
                        try:
                            val = float(text.get(key, "0"))
                            if not math.isfinite(val):
                                raise ValueError()
                            text.set(key, str(val))
                        except Exception:
                            text.set(key, "0")
                svg = ET.tostring(root, encoding="unicode")
            except Exception:
                pass
            tmp_path = EXPORT_DIR / f"__konva_page_{idx}.svg"
            tmp_path.write_text(svg, encoding="utf-8")
            tmp_paths.append(tmp_path)
            drawing = svg2rlg(str(tmp_path))
            if drawing is None:
                continue
            c.setPageSize((drawing.width, drawing.height))
            renderPDF.draw(drawing, c, 0, 0)
            c.showPage()
        c.save()
    finally:
        for p in tmp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
    return jsonify({"ok": True, "path": str(out_path), "name": out_path.name})


def _parse_color(value: str | None) -> tuple[int, int, int, int]:
    if not value:
        return (255, 255, 255, 255)
    s = value.strip().lower()
    if s.startswith("#"):
        hexv = s[1:]
        if len(hexv) == 3:
            r = int(hexv[0] * 2, 16)
            g = int(hexv[1] * 2, 16)
            b = int(hexv[2] * 2, 16)
            return (r, g, b, 255)
        if len(hexv) == 6:
            r = int(hexv[0:2], 16)
            g = int(hexv[2:4], 16)
            b = int(hexv[4:6], 16)
            return (r, g, b, 255)
        if len(hexv) == 8:
            r = int(hexv[0:2], 16)
            g = int(hexv[2:4], 16)
            b = int(hexv[4:6], 16)
            a = int(hexv[6:8], 16)
            return (r, g, b, a)
    if s.startswith("rgb"):
        nums = [int(n) for n in re.findall(r"\d+", s)[:4]]
        if len(nums) >= 3:
            r, g, b = nums[0], nums[1], nums[2]
            a = nums[3] if len(nums) >= 4 else 255
            return (r, g, b, a)
    return (255, 255, 255, 255)


def _map_get(mapping: dict | None, key: Any, default: Any = None) -> Any:
    if mapping is None:
        return default
    if key in mapping:
        return mapping[key]
    skey = str(key)
    if skey in mapping:
        return mapping[skey]
    try:
        ikey = int(key)
    except Exception:
        return default
    if ikey in mapping:
        return mapping[ikey]
    if str(ikey) in mapping:
        return mapping[str(ikey)]
    return default


def _rotate_pt(pt: list[float], angle_deg: float, cx: float, cy: float) -> list[float]:
    if not angle_deg:
        return [pt[0], pt[1]]
    ang = math.radians(angle_deg)
    c = math.cos(ang)
    s = math.sin(ang)
    x = pt[0] - cx
    y = pt[1] - cy
    return [cx + x * c - y * s, cy + x * s + y * c]


@app.post("/api/export_sim_video")
def api_export_sim_video():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    scene = payload.get("scene")
    if scene is None and SCENE_JSON.exists():
        try:
            scene = json.loads(SCENE_JSON.read_text(encoding="utf-8"))
        except Exception:
            scene = None
    if not scene:
        return jsonify({"ok": False, "error": "missing scene"}), 400
    canvas = scene.get("canvas") or {}
    w = int(canvas.get("w", 0))
    h = int(canvas.get("h", 0))
    if w <= 0 or h <= 0:
        return jsonify({"ok": False, "error": "invalid canvas"}), 400

    regions = scene.get("regions") or []
    zone_id = scene.get("zone_id") or []
    region_colors = scene.get("region_colors") or []
    zone_shift = scene.get("zone_shift") or {}
    zone_rot = scene.get("zone_rot") or {}
    zone_center = scene.get("zone_center") or {}
    zone_label_map = scene.get("zone_label_map") or {}
    zone_labels = scene.get("zone_labels") or {}
    packed_labels = payload.get("packedLabels") or []
    font_name = str(payload.get("fontName") or "Arial")
    font_size = payload.get("fontSize")
    try:
        font_size = float(font_size) * 0.5
    except Exception:
        font_size = 6.0

    zids = sorted({int(z) for z in zone_id if isinstance(z, (int, float, str))})

    def _label_for(zid: int) -> float:
        val = _map_get(zone_label_map, zid, zid)
        try:
            return float(val)
        except Exception:
            return float(zid)

    zids.sort(key=_label_for)
    zone_index = {z: idx for idx, z in enumerate(zids)}

    gap = 40
    out_w = (w * 2) + gap
    out_h = h
    fps = 6
    move_sec = 1.0
    hold_sec = 0.2
    per_zone = move_sec + hold_sec
    total_sec = max(1.0, len(zids) * per_zone)
    frame_count = max(1, int(math.ceil(total_sec * fps)))

    src_pts: list[list[list[float]]] = []
    dst_pts: list[list[list[float]]] = []
    region_zid: list[int] = []
    zone_boundaries = scene.get("zone_boundaries") or {}

    for ridx, poly in enumerate(regions):
        if not poly:
            src_pts.append([])
            dst_pts.append([])
            region_zid.append(-1)
            continue
        zid = zone_id[ridx] if ridx < len(zone_id) else -1
        region_zid.append(int(zid))
        shift = _map_get(zone_shift, zid, [0, 0])
        dx = float(shift[0]) if shift else 0.0
        dy = float(shift[1]) if shift else 0.0
        rot = float(_map_get(zone_rot, zid, 0.0) or 0.0)
        center = _map_get(zone_center, zid, [0, 0])
        cx = float(center[0]) if center else 0.0
        cy = float(center[1]) if center else 0.0
        tpts = []
        for pt in poly:
            rp = _rotate_pt(pt, rot, cx, cy)
            tpts.append([rp[0] + dx, rp[1] + dy])
        src_pts.append(tpts)
        dst_pts.append([[pt[0] + w + gap, pt[1]] for pt in poly])

    def ease_out(x: float) -> float:
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        return 1 - pow(1 - x, 3)

    def _get_font(size: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype(font_name, int(size))
        except Exception:
            return ImageFont.load_default()

    def _draw_text_center(draw: ImageDraw.ImageDraw, x: float, y: float, text: str) -> None:
        if not text:
            return
        try:
            draw.text((x, y), text, fill=(255, 255, 255, 255), font=_get_font(font_size), anchor="mm")
        except Exception:
            font = _get_font(font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x - tw / 2, y - th / 2), text, fill=(255, 255, 255, 255), font=font)

    frames: list[Image.Image] = []
    for fidx in range(frame_count):
        t = fidx / float(fps)
        img = Image.new("RGBA", (out_w, out_h), (10, 14, 28, 255))
        draw = ImageDraw.Draw(img)
        # moving index label
        active_idx = min(len(zids) - 1, max(0, int(t / per_zone))) if zids else -1
        active_zid = zids[active_idx] if active_idx >= 0 else None
        active_label = _map_get(zone_label_map, active_zid, active_zid) if active_zid is not None else "-"
        _draw_text_center(
            draw,
            out_w / 2,
            30,
            f"Moving index: {active_label}",
        )
        # right side stroke
        for zid_key, paths in zone_boundaries.items():
            try:
                zid = int(zid_key)
            except Exception:
                continue
            for pts in paths or []:
                if not pts:
                    continue
                shifted = [(p[0] + w + gap, p[1]) for p in pts]
                draw.line(shifted + [shifted[0]], fill=(245, 246, 255, 255), width=1)

        for ridx, poly in enumerate(regions):
            if not poly:
                continue
            zid = region_zid[ridx]
            idx = zone_index.get(zid, 0)
            t_rel = t - idx * per_zone
            if t_rel <= 0:
                pts = src_pts[ridx]
            elif t_rel < move_sec:
                local = ease_out(t_rel / move_sec)
                src = src_pts[ridx]
                dst = dst_pts[ridx]
                pts = [
                    [src[i][0] + (dst[i][0] - src[i][0]) * local, src[i][1] + (dst[i][1] - src[i][1]) * local]
                    for i in range(min(len(src), len(dst)))
                ]
            else:
                pts = dst_pts[ridx]
            if not pts:
                continue
            color = _parse_color(region_colors[ridx] if ridx < len(region_colors) else "#ffffff")
            draw.polygon([tuple(p) for p in pts], fill=color)

        # labels: left packed + right zone
        for lbl in packed_labels:
            try:
                lx = float(lbl.get("x", 0))
                ly = float(lbl.get("y", 0))
                text = str(lbl.get("label", ""))
            except Exception:
                continue
            _draw_text_center(draw, lx, ly, text)
        for zid_key, lbl in zone_labels.items():
            try:
                lx = float(lbl.get("x", 0)) + w + gap
                ly = float(lbl.get("y", 0))
                text = str(lbl.get("label", ""))
            except Exception:
                continue
            _draw_text_center(draw, lx, ly, text)
        frames.append(img)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPORT_DIR / f"{SVG_PATH.stem}_simulate.gif"
    try:
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
            disposal=2,
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "path": str(out_path), "name": out_path.name})


@app.get("/api/download_sim_video")
def api_download_sim_video():
    name = request.args.get("name", "")
    safe_name = os.path.basename(name)
    if not safe_name.lower().endswith(".gif"):
        return jsonify({"ok": False, "error": "invalid name"}), 400
    if not (EXPORT_DIR / safe_name).exists():
        return jsonify({"ok": False, "error": "not found"}), 404
    return send_from_directory(EXPORT_DIR, safe_name, as_attachment=True)


@app.get("/api/download_pdf")
def api_download_pdf():
    name = request.args.get("name", "")
    safe_name = os.path.basename(name)
    if not safe_name.lower().endswith(".pdf"):
        return jsonify({"ok": False, "error": "invalid name"}), 400
    if not (EXPORT_DIR / safe_name).exists():
        return jsonify({"ok": False, "error": "not found"}), 404
    return send_from_directory(EXPORT_DIR, safe_name, as_attachment=True)


@app.get("/api/download_html")
def api_download_html():
    name = request.args.get("name", "")
    safe_name = os.path.basename(name)
    if not safe_name.lower().endswith(".html"):
        return jsonify({"ok": False, "error": "invalid name"}), 400
    if not (EXPORT_DIR / safe_name).exists():
        return jsonify({"ok": False, "error": "not found"}), 404
    return send_from_directory(EXPORT_DIR, safe_name, as_attachment=True)


@app.post("/api/save_svg")
def api_save_svg():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    nodes = payload.get("nodes", [])
    segs = payload.get("segs", [])
    overlays = payload.get("overlays", []) or []
    if not SVG_PATH.exists():
        return jsonify({"ok": False, "error": "convoi.svg not found"}), 404

    if not SVG_BACKUP.exists():
        SVG_BACKUP.write_bytes(SVG_PATH.read_bytes())

    tree = ET.parse(SVG_PATH)
    root = tree.getroot()

    # remove existing line/polyline/polygon (keep image) + overlay group
    for parent in list(root.iter()):
        for child in list(parent):
            tag = child.tag.rsplit("}", 1)[-1]
            if tag in {"line", "polyline", "polygon"}:
                parent.remove(child)
            if tag == "g" and child.attrib.get("id") == "OVERLAY":
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

    if overlays:
        og = ET.Element("g", {"id": "OVERLAY"})
        for item in overlays:
            try:
                src = str(item.get("src") or "")
                if not src:
                    continue
                x = float(item.get("x") or 0)
                y = float(item.get("y") or 0)
                w = float(item.get("width") or 0) or 1.0
                h = float(item.get("height") or 0) or 1.0
                sx = float(item.get("scaleX") or 1.0)
                sy = float(item.get("scaleY") or 1.0)
                rot = float(item.get("rotation") or 0.0)
            except Exception:
                continue
            img = ET.Element(
                "image",
                {
                    "x": str(-w / 2.0),
                    "y": str(-h / 2.0),
                    "width": str(w),
                    "height": str(h),
                    "transform": f"translate({x} {y}) rotate({rot}) scale({sx} {sy})",
                    "data-overlay": "1",
                    "data-id": str(item.get("id") or ""),
                    "data-x": str(x),
                    "data-y": str(y),
                    "data-width": str(w),
                    "data-height": str(h),
                    "data-scale-x": str(sx),
                    "data-scale-y": str(sy),
                    "data-rotation": str(rot),
                },
            )
            img.set("{http://www.w3.org/1999/xlink}href", src)
            img.set("href", src)
            og.append(img)
        root.append(og)

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
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        ensure_outputs()
    app.run(host="127.0.0.1", port=5000, debug=True)
