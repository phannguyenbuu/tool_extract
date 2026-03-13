from __future__ import annotations

import json
import os
import subprocess
import math
import re
import time
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

import new_toy
import packing, zones, config, source_voronoi

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
WEB_DIR = ROOT / "frontend"
DIST_DIR = WEB_DIR / "dist"
SOURCES_DIR = ROOT / "sources"
ACTIVE_SOURCE_JSON = BACKEND_DIR / "active_source.json"
DEFAULT_SOURCE_NAME = "convoi.svg"
LEGACY_STATE_JSON = ROOT / "scripts" / "ui_state.json"
LEGACY_STATE_SVG = ROOT / "scripts" / "ui_state.svg"
LEGACY_PACKED_LABELS_JSON = ROOT / "scripts" / "packed_labels.json"
LEGACY_ZONE_LABELS_JSON = ROOT / "scripts" / "zone_labels.json"
LEGACY_SCENE_JSON = ROOT / "scripts" / "scene_cache.json"
LEGACY_SOURCE_ZONE_CLICK_JSON = ROOT / "scripts" / "source_zone_click.json"
LEGACY_SOURCE_ZONE_CLICK_JSON_OLD = ROOT / "scripts" / "soure_zone_click.json"
PACKED_ZONE_SCENE_SVG = ROOT / "scripts" / "packed_zone_scene.svg"
PACKED_ZONE_SCENE_SVG_PAGE2 = ROOT / "scripts" / "packed_zone_scene_page2.svg"
RASTER_PACK_TMP_JSON = ROOT / "scripts" / "tmp_raster_pack.json"
RASTER_PACK_TMP_PNG = ROOT / "scripts" / "tmp_raster_pack.png"
EXPORT_DIR = ROOT / "export"
# Show bleed layer in packed output.
PACKED_INCLUDE_BLEED = True

app = Flask(__name__, static_folder=None)


def _cache_file(prefix: str, source_name: str, ext: str = ".json") -> Path:
    stem = Path(source_name).stem
    return ROOT / "scripts" / f"{prefix}_{stem}{ext}"


def _state_json_path(source_name: str) -> Path:
    return _cache_file("ui_state", source_name)


def _state_svg_path(source_name: str) -> Path:
    return _cache_file("ui_state", source_name, ext=".svg")


def _packed_labels_path(source_name: str) -> Path:
    return _cache_file("packed_labels", source_name)


def _zone_labels_path(source_name: str) -> Path:
    return _cache_file("zone_labels", source_name)


def _scene_cache_path(source_name: str) -> Path:
    return _cache_file("scene_cache", source_name)


def _source_click_path(source_name: str) -> Path:
    return _cache_file("source_zone_click", source_name)


def _source_snap_region_map_path(source_name: str) -> Path:
    return _cache_file("source_snap_region_map", source_name)


def _zone_debug_path(source_name: str) -> Path:
    return _cache_file("zone_debug", source_name)


def _source_edit_cache_path(source_name: str) -> Path:
    return _cache_file("source_edit_cache", source_name)


def _legacy_fallback_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_state_json(source_name: str) -> dict:
    state_json = _state_json_path(source_name)
    if state_json.exists():
        try:
            return json.loads(state_json.read_text(encoding="utf-8"))
        except Exception:
            return {}
    if source_name == DEFAULT_SOURCE_NAME and LEGACY_STATE_JSON.exists():
        return _legacy_fallback_json(LEGACY_STATE_JSON)
    return {}


def _load_source_edit_cache(source_name: str) -> dict:
    cache_path = _source_edit_cache_path(source_name)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _strip_source_edit_keys(data: dict | None) -> dict:
    if not isinstance(data, dict):
        return {}
    out = dict(data)
    for key in ("svg_nodes", "svg_segments", "source_voronoi"):
        out.pop(key, None)
    return out


def _list_recent_source_names() -> list[str]:
    names: list[str] = []
    if SOURCES_DIR.exists():
        for p in sorted(SOURCES_DIR.glob("*.svg"), key=lambda x: x.name.lower()):
            if p.name.lower().endswith("_fill.svg"):
                continue
            if p.name not in names:
                names.append(p.name)
    return names


def _resolve_source_name(name: str | None) -> str | None:
    if not name:
        return None
    safe = os.path.basename(str(name).strip())
    if not safe.lower().endswith(".svg"):
        return None
    if safe in _list_recent_source_names():
        return safe
    return None


def _source_path_from_name(source_name: str) -> Path | None:
    if source_name == DEFAULT_SOURCE_NAME and (ROOT / source_name).exists():
        return ROOT / source_name
    p = SOURCES_DIR / source_name
    if p.exists():
        return p
    p2 = ROOT / source_name
    if p2.exists():
        return p2
    return None


def _set_active_source_name(source_name: str) -> None:
    ACTIVE_SOURCE_JSON.write_text(
        json.dumps({"name": source_name}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _get_active_source_name(requested: str | None = None) -> str:
    resolved = _resolve_source_name(requested)
    if resolved:
        _set_active_source_name(resolved)
        return resolved
    if ACTIVE_SOURCE_JSON.exists():
        try:
            data = json.loads(ACTIVE_SOURCE_JSON.read_text(encoding="utf-8"))
            cached = _resolve_source_name(data.get("name"))
            if cached:
                return cached
        except Exception:
            pass
    files = _list_recent_source_names()
    source_name = files[0] if files else DEFAULT_SOURCE_NAME
    _set_active_source_name(source_name)
    return source_name


def _activate_source(source_name: str) -> Path:
    src = _source_path_from_name(source_name)
    if src is None:
        raise FileNotFoundError(f"source not found: {source_name}")
    config.SVG_PATH = src
    new_toy.config.SVG_PATH = src
    try:
        setattr(new_toy, "SVG_PATH", src)
    except Exception:
        pass
    return src


@app.get("/api/recent_sources")
def api_recent_sources():
    files = _list_recent_source_names()
    active = _get_active_source_name(request.args.get("source"))
    return jsonify({"files": files, "active": active})


def ensure_outputs() -> None:
    env = os.environ.copy()
    env["INTERSECT_SNAP"] = str(new_toy.INTERSECT_SNAP)
    env["LINE_EXTEND"] = str(new_toy.LINE_EXTEND)
    cmd = [os.fspath(Path(os.environ.get("PYTHON", "python"))), os.fspath(BACKEND_DIR / "new_toy.py")]
    subprocess.run(cmd, cwd=BACKEND_DIR, env=env, check=False)


@app.get("/api/scene")
def api_scene():
    snap = request.args.get("snap", type=float) or new_toy.INTERSECT_SNAP
    source_name = _get_active_source_name(request.args.get("source"))
    source_path = _activate_source(source_name)
    # YOLO MODE: FORCE RECOMPUTE
    scene_json = _scene_cache_path(source_name)
    if False and scene_json.exists():
        return json.loads(scene_json.read_text(encoding="utf-8"))
    zone_labels_json = _zone_labels_path(source_name)
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

    data = new_toy.compute_scene(source_path, snap, render_packed_png=False)
    data["source_name"] = source_name
    try:
        scene_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if source_name == DEFAULT_SOURCE_NAME:
            LEGACY_SCENE_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        zone_labels = data.get("zone_labels")
        if isinstance(zone_labels, dict):
            zone_labels_json.write_text(
                json.dumps(zone_labels, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if source_name == DEFAULT_SOURCE_NAME:
                LEGACY_ZONE_LABELS_JSON.write_text(
                    json.dumps(zone_labels, ensure_ascii=False, indent=2), encoding="utf-8"
                )
    except Exception:
        pass
    # Do not generate packed preview during /api/scene reload.
    # Packed output is generated only via /api/pack_from_scene (Compute/Repack).
    return jsonify(data)


@app.get("/api/zone_debug")
def api_zone_debug():
    source_name = _get_active_source_name(request.args.get("source"))
    path = _zone_debug_path(source_name)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return jsonify({"vertices": {k: v.get("vertices") for k, v in data.items()}})
        except Exception:
            pass
    return jsonify({"vertices": {}, "error": "missing zone_debug"})


@app.get("/api/source_voronoi")
def api_source_voronoi():
    source_name = _get_active_source_name(request.args.get("source"))
    source_path = _activate_source(source_name)
    cached = _load_source_edit_cache(source_name).get("source_voronoi")
    if isinstance(cached, dict):
        data = dict(cached)
        if "snappedCells" in data and "snapped_cells" not in data:
            data["snapped_cells"] = data.get("snappedCells") or []
        data["source_name"] = source_name
        return jsonify(data)
    count = request.args.get("count", type=int) or config.TARGET_ZONES
    relax = request.args.get("relax", type=int)
    seed = request.args.get("seed", type=int)
    data = source_voronoi.build_source_voronoi(
        source_path,
        count=count,
        relax=2 if relax is None else relax,
        seed=7 if seed is None else seed,
    )
    data["source_name"] = source_name
    return jsonify(data)


@app.get("/api/source_region_scene")
def api_source_region_scene():
    source_name = _get_active_source_name(request.args.get("source"))
    source_path = _activate_source(source_name)
    count = request.args.get("count", type=int) or config.TARGET_ZONES
    relax = request.args.get("relax", type=int)
    seed = request.args.get("seed", type=int)
    source_edit = _load_source_edit_cache(source_name)
    data = source_voronoi.build_source_region_scene(
        source_path,
        count=count,
        relax=2 if relax is None else relax,
        seed=7 if seed is None else seed,
        cached_nodes=source_edit.get("svg_nodes"),
        cached_segments=source_edit.get("svg_segments"),
        cached_voronoi=source_edit.get("source_voronoi"),
    )
    data["source_name"] = source_name
    try:
        _source_snap_region_map_path(source_name).write_text(
            json.dumps(
                {
                    "source_name": source_name,
                    "snap_region_map": data.get("snap_region_map", {}),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
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
    source_name = _get_active_source_name(request.args.get("source"))
    state_json = _state_json_path(source_name)
    state_svg = _state_svg_path(source_name)
    source_edit_json = _source_edit_cache_path(source_name)
    state_payload = _strip_source_edit_keys(payload)
    state_json.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if source_name == DEFAULT_SOURCE_NAME:
        LEGACY_STATE_JSON.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    source_edit_payload = {}
    for key in ("svg_nodes", "svg_segments", "source_voronoi"):
        if key in payload:
            source_edit_payload[key] = payload.get(key)
    if source_edit_payload:
        source_edit_json.write_text(
            json.dumps(source_edit_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # build a lightweight svg for fast restore
    canvas = state_payload.get("canvas", {})
    w = canvas.get("w", 1000)
    h = canvas.get("h", 1000)
    paths = []
    for region in state_payload.get("regions", []):
        if not region:
            continue
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in region) + " Z"
        paths.append(f'<path d="{d}" fill="none" stroke="#999" stroke-width="0.5"/>')
    labels = []
    for lbl in state_payload.get("labels", []):
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
    state_svg.write_text(svg, encoding="utf-8")
    if source_name == DEFAULT_SOURCE_NAME:
        LEGACY_STATE_SVG.write_text(svg, encoding="utf-8")
    return jsonify({"ok": True})


@app.post("/api/reset_source_cache")
def api_reset_source_cache():
    source_name = _get_active_source_name(request.args.get("source"))
    edit_cache = _source_edit_cache_path(source_name)
    snap_map = _source_snap_region_map_path(source_name)
    for path in (edit_cache, snap_map):
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

    state_json = _state_json_path(source_name)
    if state_json.exists():
        try:
            data = json.loads(state_json.read_text(encoding="utf-8"))
            data.pop("source_region_scene_cache", None)
            state_json.write_text(
                json.dumps(_strip_source_edit_keys(data), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
    if source_name == DEFAULT_SOURCE_NAME and LEGACY_STATE_JSON.exists():
        try:
            data = json.loads(LEGACY_STATE_JSON.read_text(encoding="utf-8"))
            data.pop("source_region_scene_cache", None)
            LEGACY_STATE_JSON.write_text(
                json.dumps(_strip_source_edit_keys(data), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
    return jsonify({"ok": True})


@app.get("/api/state")
def api_get_state():
    source_name = _get_active_source_name(request.args.get("source"))
    data = _load_state_json(source_name)
    data.update(_load_source_edit_cache(source_name))
    return jsonify(data)


@app.get("/api/packed_labels")
def api_packed_labels():
    source_name = _get_active_source_name(request.args.get("source"))
    packed_labels_json = _packed_labels_path(source_name)
    if packed_labels_json.exists():
        try:
            data = json.loads(packed_labels_json.read_text(encoding="utf-8"))
            return jsonify(data)
        except Exception:
            return jsonify({})
    if source_name == DEFAULT_SOURCE_NAME and LEGACY_PACKED_LABELS_JSON.exists():
        return jsonify(_legacy_fallback_json(LEGACY_PACKED_LABELS_JSON))
    return jsonify({})


@app.post("/api/packed_labels")
def api_save_packed_labels():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    source_name = _get_active_source_name(request.args.get("source"))
    packed_labels_json = _packed_labels_path(source_name)
    data: Dict[str, Any] = {}
    if packed_labels_json.exists():
        try:
            data = json.loads(packed_labels_json.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    for key, val in payload.items():
        data[str(key)] = val
    packed_labels_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if source_name == DEFAULT_SOURCE_NAME:
        LEGACY_PACKED_LABELS_JSON.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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
    t_total_start = time.perf_counter()
    timings_ms: Dict[str, float] = {}
    _t = t_total_start
    def _mark(name: str) -> None:
        nonlocal _t
        now = time.perf_counter()
        timings_ms[name] = round((now - _t) * 1000.0, 2)
        _t = now

    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    source_name = _get_active_source_name(request.args.get("source") or payload.get("source"))
    _activate_source(source_name)
    raster_only = bool(payload.get("raster_only", False))
    print(f"[pack_from_scene] start source={source_name} raster_only={int(raster_only)}")
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
        if key in payload and payload.get(key) is not None:
            os.environ[env_key] = str(payload.get(key))
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
    unique_zone_count = len({int(z) for z in zone_id if isinstance(z, (int, float, str))})
    
    # Cleanup temp files to prevent stale reads
    for p in [RASTER_PACK_TMP_JSON, config.OUT_PACK_SVG, config.OUT_PACK_SVG_PAGE2]:
        try:
            if p.exists():
                os.remove(p)
        except Exception:
            pass

    print(
        "[pack_from_scene] input "
        f"canvas={w}x{h} regions={len(regions)} zones={unique_zone_count} "
        f"padding={getattr(config, 'PADDING', 0)} bleed={getattr(config, 'PACK_BLEED', 0)} "
        f"grid={getattr(config, 'PACK_GRID_STEP', 0)} angle={getattr(config, 'PACK_ANGLE_STEP', 0)} "
        f"mode={getattr(config, 'PACK_MODE', 'fast')}"
    )
    _mark("init_validate")
    grid_step = max(1.0, float(getattr(config, "PACK_GRID_STEP", 5.0) or 5.0))
    # Fixed margin as requested.
    config.PACK_MARGIN_X = 10
    config.PACK_MARGIN_Y = 10
    # Always compute fresh (no cache), using TS-style raster-first packing.
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
        polys.append(pts if pts else [])
    _mark("normalize_regions")
    zone_polys_payload = payload.get("zone_polys") or []
    zone_order_payload = payload.get("zone_order") or []
    zone_polys: list[list[tuple[float, float]]] = []
    for poly in zone_polys_payload:
        pts = []
        for p in poly or []:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                pts.append((float(p[0]), float(p[1])))
            except Exception:
                continue
        if len(pts) >= 3:
            zone_polys.append(pts)
    explicit_zone_polys = bool(zone_polys)
    if explicit_zone_polys:
        zone_order = []
        for idx, zid in enumerate(zone_order_payload):
            try:
                zone_order.append(int(zid))
            except Exception:
                zone_order.append(idx)
        if len(zone_order) != len(zone_polys):
            zone_order = list(range(len(zone_polys)))
    else:
        zone_polys, zone_order, _zone_poly_debug = zones.build_zone_polys(polys, zone_id)
    print(
        f"[pack_from_scene] zone_polys built={len(zone_polys)} zone_order={len(zone_order)} "
        f"explicit={int(explicit_zone_polys)}"
    )
    _mark("build_zone_polys")
    # Bleed is applied before raster nest via inflated zone polygons.
    zone_pack_polys = packing._build_zone_pack_polys(
        zone_polys, float(config.PACK_BLEED), bevel_angle=60.0
    )
    print(
        "[pack_from_scene] bleed_polys "
        f"enabled={int(float(getattr(config, 'PACK_BLEED', 0) or 0) > 0)} "
        f"bleed={float(getattr(config, 'PACK_BLEED', 0) or 0):.2f}"
    )
    _mark("build_bleed_polys")
    # TS-style raster pack (ultra-fast): large-first, then fill holes with small pieces.
    margin_x = max(0.0, float(getattr(config, "PACK_MARGIN_X", 0.0) or 0.0))
    margin_y = max(0.0, float(getattr(config, "PACK_MARGIN_Y", 0.0) or 0.0))
    pack_gap = max(0.0, float(getattr(config, "PADDING", 0.0) or 0.0))
    n = len(zone_pack_polys)
    placements: list[tuple[float, float, int, int, bool]] = [(-1.0, -1.0, 0, 0, False)] * n
    rot_info: list[dict[str, float]] = [
        {"angle": 0.0, "cx": 0.0, "cy": 0.0, "minx": 0.0, "miny": 0.0, "bin": -1}
        for _ in range(n)
    ]
    raster_report: Dict[str, Any] = {"count": 0, "pairs": []}
    raster_scale_factor = 4.0  # 1/4 resolution
    # Denser setting to avoid sparse layout and overflow artifacts.
    grid_for_raster = 6.0
    raster_cell = max(4, int(round(min(12.0, grid_for_raster))))
    search_stride = 2
    safety_for_raster = max(0.25, min(1.0, pack_gap * 0.15))
    # 5-degree step full half-turn for better local orientation near dense rows.
    rotations_5 = [float(a) for a in range(0, 180, 5)]
    print(
        "[pack_from_scene] raster_config "
        f"scale=1/{int(raster_scale_factor)} grid_for_raster={grid_for_raster} "
        f"cell={raster_cell} stride={search_stride} safety={safety_for_raster:.3f} "
        f"rotations={len(rotations_5)}"
    )

    zone_pack_centers = []
    area_map: Dict[int, float] = {}
    for idx, p in enumerate(zone_pack_polys):
        if p:
            pg = packing.Polygon(p)
            if pg.is_empty:
                zone_pack_centers.append((0.0, 0.0))
                area_map[idx] = 0.0
            else:
                c = pg.centroid
                zone_pack_centers.append((float(c.x), float(c.y)))
                area_map[idx] = abs(float(pg.area))
        else:
            zone_pack_centers.append((0.0, 0.0))
            area_map[idx] = 0.0
    _mark("prep_centers_area")

    sorted_desc = sorted(range(n), key=lambda i: area_map.get(i, 0.0), reverse=True)
    split_large = max(1, int(round(len(sorted_desc) * 0.55))) if sorted_desc else 0
    large_ids = sorted_desc[:split_large]
    small_ids = sorted_desc[split_large:]
    small_asc = sorted(small_ids, key=lambda i: area_map.get(i, 0.0))
    # TS-like queue: place large first, then let small pieces fill holes.
    queue = large_ids + small_asc
    print(
        "[pack_from_scene] queue "
        f"total={len(queue)} large_first={len(large_ids)} small_fill={len(small_asc)}"
    )

    # One-page TS-like queue pass.
    pp, _o, rr = packing.pack_regions_raster_fast(
        zone_pack_polys,
        (w, h),
        fixed_centers=zone_pack_centers,
        grid_step=grid_for_raster,
        rotations=rotations_5,
        search_stride=search_stride,
        safety_padding=safety_for_raster,
        place_ids=queue,
    )
    placed_now: set[int] = set()
    for idx in queue:
        if idx >= len(pp) or idx >= len(rr):
            continue
        dx, dy, bw, bh, rf = pp[idx]
        if bw <= 0 or bh <= 0:
            continue
        placements[idx] = (dx, dy, bw, bh, rf)
        ri = dict(rr[idx])
        ri["bin"] = 0
        rot_info[idx] = ri
        placed_now.add(idx)
    queue = [idx for idx in queue if idx not in placed_now]
    print(
        "[pack_from_scene] page1 "
        f"placed={len(placed_now)} unplaced={len(queue)} "
        f"fill_rate={(len(placed_now) / max(1, n)):.3f}"
    )
    _mark("pack_page1")

    # Center packed bbox in canvas for cleaner margins.
    placed_ids = [i for i in range(n) if placements[i][2] > 0 and placements[i][3] > 0]
    if placed_ids:
        minx = miny = float("inf")
        maxx = maxy = float("-inf")
        for rid in placed_ids:
            pts = zone_pack_polys[rid]
            if not pts:
                continue
            info = rot_info[rid]
            dx, dy, _, _, _ = placements[rid]
            ang = float(info.get("angle", 0.0))
            cx = float(info.get("cx", 0.0))
            cy = float(info.get("cy", 0.0))
            for p in packing._rotate_pts(pts, ang, cx, cy):
                x = float(p[0]) + float(dx)
                y = float(p[1]) + float(dy)
                minx = min(minx, x)
                miny = min(miny, y)
                maxx = max(maxx, x)
                maxy = max(maxy, y)
        if minx < maxx and miny < maxy and math.isfinite(minx) and math.isfinite(maxx):
            cx_box = 0.5 * (minx + maxx)
            cy_box = 0.5 * (miny + maxy)
            shift_x = (float(w) * 0.5) - cx_box
            shift_y = (float(h) * 0.5) - cy_box
            for rid in placed_ids:
                dx, dy, bw, bh, rf = placements[rid]
                placements[rid] = (float(dx) + shift_x, float(dy) + shift_y, bw, bh, rf)
            print(
                "[pack_from_scene] center_bbox "
                f"placed={len(placed_ids)} shift=({shift_x:.2f},{shift_y:.2f}) "
                f"bbox=({minx:.2f},{miny:.2f})-({maxx:.2f},{maxy:.2f})"
            )
    else:
        print("[pack_from_scene] center_bbox skipped=no placed zones")
    _mark("center_bbox")

    unplaced_count = len(queue)
    if unplaced_count > 0:
        print(f"[pack_from_scene] WARNING: one page overflow, unplaced={unplaced_count}")
    else:
        print("[pack_from_scene] One page OK")

    raster_report = packing.raster_overlap_report(
        zone_pack_polys, placements, rot_info, (w, h), cell=raster_cell
    )
    overlap_pairs = raster_report.get("pairs", []) or []
    print(
        "[pack_from_scene] raster_overlap "
        f"count={int(raster_report.get('count', 0))} "
        f"sample_pairs={overlap_pairs[:5]}"
    )
    _mark("raster_overlap")

    tmp_payload = {
        "canvas": {"w": w, "h": h},
        "pages": 1,
        "raster_scale_factor": raster_scale_factor,
        "strict_gap": pack_gap,
        "one_page_ok": unplaced_count == 0,
        "unplaced_count": unplaced_count,
        "timings_ms": timings_ms,
        "raster_check": raster_report,
        "placements": [[float(a), float(b), int(c), int(d), bool(e)] for (a, b, c, d, e) in placements],
        "rot_info": rot_info,
    }
    # Removed RASTER_PACK_TMP_JSON generation
    _mark("skip_tmp_json")
    try:
        img = Image.new("RGB", (max(1, int(w)), max(1, int(h))), (6, 14, 46))
        draw = ImageDraw.Draw(img, "RGBA")
        overlap_ids = set()
        for pair in raster_report.get("pairs", []) or []:
            try:
                overlap_ids.add(int(pair[0]))
                overlap_ids.add(int(pair[1]))
            except Exception:
                continue
        for rid, pts in enumerate(zone_pack_polys):
            if rid >= len(placements) or rid >= len(rot_info) or not pts:
                continue
            dx, dy, bw, bh, _ = placements[rid]
            if bw <= 0 or bh <= 0:
                continue
            info = rot_info[rid]
            bin_idx = int(info.get("bin", 0))
            if bin_idx != 0:
                continue
            try:
                ang = float(info.get("angle", 0.0))
                cx = float(info.get("cx", 0.0))
                cy = float(info.get("cy", 0.0))
            except Exception:
                continue
            tpts = []
            for p in packing._rotate_pts(pts, ang, cx, cy):
                x = float(p[0]) + float(dx)
                y = float(p[1]) + float(dy)
                if not (math.isfinite(x) and math.isfinite(y)):
                    tpts = []
                    break
                tpts.append((x, y))
            if len(tpts) < 3:
                continue
            r = (rid * 67) % 180 + 60
            g = (rid * 41) % 180 + 60
            b = (rid * 23) % 180 + 60
            try:
                draw.polygon(tpts, fill=(r, g, b, 130))
                if rid in overlap_ids:
                    draw.line(tpts + [tpts[0]], fill=(255, 64, 64, 255), width=3)
                else:
                    draw.line(tpts + [tpts[0]], fill=(255, 255, 255, 220), width=1)
            except Exception:
                continue
        img.save(RASTER_PACK_TMP_PNG)
        print(f"[pack_from_scene] write tmp_png path={RASTER_PACK_TMP_PNG}")
    except Exception:
        pass
    _mark("write_tmp_png")
    placement_bin = [int(info.get("bin", -1)) for info in rot_info]
    placement_bin_by_zid = {}
    for idx, zid in enumerate(zone_order):
        if idx >= len(placement_bin) or idx >= len(placements):
            continue
        if placements[idx][2] <= 0 or placements[idx][3] <= 0:
            continue
        placement_bin_by_zid[zid] = placement_bin[idx]
    zone_shift: Dict[int, tuple[float, float]] = {}
    zone_rot: Dict[int, float] = {}
    zone_center: Dict[int, tuple[float, float]] = {}
    for idx, zid in enumerate(zone_order):
        if idx >= len(placements):
            continue
        dx, dy, bw, bh, _ = placements[idx]
        if bw <= 0 or bh <= 0:
            continue
        zone_shift[zid] = (float(dx), float(dy))
        info = rot_info[idx] if idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
        zone_rot[zid] = float(info.get("angle", 0.0))
        zone_center[zid] = (float(info.get("cx", 0.0)), float(info.get("cy", 0.0)))
    if raster_only:
        timings_ms["total"] = round((time.perf_counter() - t_total_start) * 1000.0, 2)
        print(f"[pack_from_scene] done raster_only=1 timings_ms={timings_ms}")
        return jsonify(
            {
                "ok": True,
                "raster_only": True,
                "raster_tmp_path": str(RASTER_PACK_TMP_PNG),
                "raster_tmp_png_path": str(RASTER_PACK_TMP_PNG),
                "raster_tmp_png_url": "/out/tmp_raster_pack.png",
                "raster_tmp_json_path": str(RASTER_PACK_TMP_JSON),
                "raster_overlap_count": int(raster_report.get("count", 0)),
                "raster_pages": 1,
                "one_page_ok": unplaced_count == 0,
                "unplaced_count": unplaced_count,
                "timings_ms": timings_ms,
            }
        )
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
        include_bleed=PACKED_INCLUDE_BLEED,
    )
    print(
        "[pack_from_scene] write_svg "
        f"path={PACKED_ZONE_SCENE_SVG} include_bleed={int(bool(PACKED_INCLUDE_BLEED))}"
    )
    _mark("write_svg_page1")
    timings_ms["total"] = round((time.perf_counter() - t_total_start) * 1000.0, 2)
    print(
        "[pack_from_scene] done "
        f"one_page_ok={int(unplaced_count == 0)} unplaced={unplaced_count} "
        f"timings_ms={timings_ms}"
    )
    result = {
        "ok": True,
        "source_name": source_name,
        "packed_svg": PACKED_ZONE_SCENE_SVG.read_text(encoding="utf-8"),
        "packed_svg_page2": "",
        "zone_polys": zone_polys,
        "zone_order": zone_order,
        "zone_pack_polys": zone_pack_polys,
        "zone_shift": zone_shift,
        "zone_rot": zone_rot,
        "zone_center": zone_center,
        "placement_bin": placement_bin_by_zid,
        "raster_tmp_path": str(RASTER_PACK_TMP_PNG),
        "raster_tmp_png_path": str(RASTER_PACK_TMP_PNG),
        "raster_tmp_png_url": "/out/tmp_raster_pack.png",
        "raster_tmp_json_path": str(RASTER_PACK_TMP_JSON),
        "raster_overlap_count": int(raster_report.get("count", 0)),
        "raster_pages": 1,
        "one_page_ok": unplaced_count == 0,
        "unplaced_count": unplaced_count,
        "timings_ms": timings_ms,
    }
    return jsonify(result)


@app.get("/api/source_zone_click")
def api_get_source_zone_click():
    source_name = _get_active_source_name(request.args.get("source"))
    src = _source_click_path(source_name)
    if not src.exists() and source_name == DEFAULT_SOURCE_NAME:
        src = (
            LEGACY_SOURCE_ZONE_CLICK_JSON
            if LEGACY_SOURCE_ZONE_CLICK_JSON.exists()
            else LEGACY_SOURCE_ZONE_CLICK_JSON_OLD
        )
    if src.exists():
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
            if isinstance(data, list):
                print(f"[source_zone_click] load source={source_name} path={src} count={len(data)}")
                return jsonify({"clicks": data})
            if isinstance(data, dict):
                clicks = data.get("clicks", [])
                print(f"[source_zone_click] load source={source_name} path={src} count={len(clicks)}")
                return jsonify({"clicks": clicks})
        except Exception:
            print(f"[source_zone_click] load source={source_name} path={src} parse_error=1")
            return jsonify({"clicks": []})
    print(f"[source_zone_click] load source={source_name} path={src} count=0")
    return jsonify({"clicks": []})


@app.post("/api/source_zone_click")
def api_save_source_zone_click():
    source_name = _get_active_source_name(request.args.get("source"))
    payload: Any = request.get_json(silent=True) or []
    clicks = payload if isinstance(payload, list) else payload.get("clicks", [])
    cleaned = []
    for item in clicks or []:
        if not isinstance(item, dict):
            continue
        x = item.get("x")
        y = item.get("y")
        rid = item.get("rid")
        attach_to = item.get("attach_to")
        row: Dict[str, Any] = {}
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if math.isfinite(x) and math.isfinite(y):
                row["x"] = float(x)
                row["y"] = float(y)
        if isinstance(rid, (int, float)) and math.isfinite(float(rid)):
            row["rid"] = int(rid)
        if isinstance(attach_to, (int, float)) and math.isfinite(float(attach_to)):
            row["attach_to"] = int(attach_to)
        if row:
            cleaned.append(row)
    payload_text = json.dumps({"clicks": cleaned}, ensure_ascii=False, indent=2)
    _source_click_path(source_name).write_text(payload_text, encoding="utf-8")
    print(
        f"[source_zone_click] save source={source_name} path={_source_click_path(source_name)} "
        f"count={len(cleaned)}"
    )
    if source_name == DEFAULT_SOURCE_NAME:
        try:
            LEGACY_SOURCE_ZONE_CLICK_JSON.write_text(payload_text, encoding="utf-8")
            LEGACY_SOURCE_ZONE_CLICK_JSON_OLD.write_text(payload_text, encoding="utf-8")
        except Exception:
            pass
    return jsonify({"ok": True, "count": len(cleaned)})


@app.post("/api/export")
def api_export():
    try:
        source_name = _get_active_source_name(request.args.get("source"))
        src_path = _activate_source(source_name)
        scene_json = _scene_cache_path(source_name)
        zone_labels_json = _zone_labels_path(source_name)
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
        if scene_json.exists():
            try:
                cached_scene = json.loads(scene_json.read_text(encoding="utf-8"))
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

        prefix = src_path.stem
        print("[export] 90% write svgs")
        prefix = src_path.stem
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

            def zone_stroke_color(zid):
                return "#ffffff"

            def zone_stroke_width(zid):
                return "1"

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
                    parts.append(
                        f'<path d="{d}" fill="none" stroke="{zone_stroke_color(zid)}" stroke-width="{zone_stroke_width(zid)}"/>'
                    )
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
                    parts.append(
                        f'<path d="{d}" fill="none" stroke="{zone_stroke_color(zid)}" stroke-width="{zone_stroke_width(zid)}"/>'
                    )
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
            if zone_labels_json.exists():
                try:
                    labels = json.loads(zone_labels_json.read_text(encoding="utf-8"))
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
    source_name = _get_active_source_name(request.args.get("source"))
    src_path = _activate_source(source_name)
    prefix = f"{src_path.stem}_"
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
    source_name = _get_active_source_name(request.args.get("source") or payload.get("source"))
    src_path = _activate_source(source_name)
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
    except Exception as e:
        try:
            import site
            import sys

            user_site = site.getusersitepackages()
            if user_site and user_site not in sys.path:
                sys.path.append(user_site)
            from reportlab.pdfgen import canvas as pdf_canvas
            from reportlab.graphics import renderPDF
            from svglib.svglib import svg2rlg
        except Exception as inner:
            try:
                import site
                import sys

                detail = {
                    "executable": sys.executable,
                    "user_site": site.getusersitepackages(),
                    "path": sys.path,
                }
            except Exception:
                detail = {}
            return jsonify(
                {
                    "ok": False,
                    "error": f"Missing reportlab/svglib: {repr(inner)}",
                    "detail": detail,
                }
            ), 500

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPORT_DIR / f"{src_path.stem}_konva.pdf"
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
    source_name = _get_active_source_name(request.args.get("source") or payload.get("source"))
    src_path = _activate_source(source_name)
    scene_json = _scene_cache_path(source_name)
    scene = payload.get("scene")
    if scene is None and scene_json.exists():
        try:
            scene = json.loads(scene_json.read_text(encoding="utf-8"))
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
    out_path = EXPORT_DIR / f"{src_path.stem}_simulate.gif"
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
    source_name = _get_active_source_name(request.args.get("source") or payload.get("source"))
    svg_path = _activate_source(source_name)
    svg_backup = ROOT / f"{svg_path.stem}_backup.svg"
    nodes = payload.get("nodes", [])
    segs = payload.get("segs", [])
    overlays = payload.get("overlays", []) or []
    if not svg_path.exists():
        return jsonify({"ok": False, "error": f"{source_name} not found"}), 404

    if not svg_backup.exists():
        svg_backup.write_bytes(svg_path.read_bytes())

    tree = ET.parse(svg_path)
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

    tree.write(svg_path, encoding="utf-8", xml_declaration=True)
    return jsonify({"ok": True})


@app.get("/out/<path:path>")
def output_files(path: str):
    # 1. Try directly from ROOT (e.g. /out/sources/...)
    if (ROOT / path).exists():
        return send_from_directory(ROOT, path)
    # 2. Try from scripts/ (e.g. /out/tmp_raster_pack.png)
    if (ROOT / "scripts" / path).exists():
        return send_from_directory(ROOT / "scripts", path)
    # 3. Handle convoi.svg specially if requested at root
    if path == DEFAULT_SOURCE_NAME:
        if (SOURCES_DIR / DEFAULT_SOURCE_NAME).exists():
            return send_from_directory(SOURCES_DIR, DEFAULT_SOURCE_NAME)
    return "Not found", 404


@app.get("/")
def index():
    if DIST_DIR.exists():
        return send_from_directory(DIST_DIR, "index.html")
    return "Frontend not built. Run 'npm run build' in frontend directory.", 404


@app.get("/<path:path>")
def serve_static(path):
    if DIST_DIR.exists() and (DIST_DIR / path).exists():
        return send_from_directory(DIST_DIR, path)
    if DIST_DIR.exists():
        return send_from_directory(DIST_DIR, "index.html")
    return "Not found", 404


if __name__ == "__main__":
    server_debug = str(os.environ.get("SERVER_DEBUG", "1")).strip().lower() not in {"0", "false", "no"}
    use_reloader = str(os.environ.get("SERVER_RELOADER", "1")).strip().lower() not in {"0", "false", "no"}
    if server_debug and os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        ensure_outputs()
    app.run(host="127.0.0.1", port=5000, debug=server_debug, use_reloader=use_reloader)
