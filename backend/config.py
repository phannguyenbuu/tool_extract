from __future__ import annotations

import os
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"

SVG_PATH = ROOT / "sources" / "convoi.svg"
OUT_LOG = SCRIPTS_DIR / "regions_log.txt"
OUT_PNG = SCRIPTS_DIR / "regions_log.png"
OUT_ZONES_LOG = SCRIPTS_DIR / "zones_log.txt"
OUT_ZONES_PNG = SCRIPTS_DIR / "zones_log.png"
OUT_PACK_LOG = SCRIPTS_DIR / "packed_log.txt"
OUT_PACK_PNG = SCRIPTS_DIR / "packed.png"
OUT_PACK_OUTLINE_PNG = SCRIPTS_DIR / "packed_outline.png"
OUT_PACK_SVG = SCRIPTS_DIR / "packed.svg"
OUT_PACK_BBOX_SVG = SCRIPTS_DIR / "packed_bbox.svg"
OUT_PACK_BBOX_VS_POLY_SVG = SCRIPTS_DIR / "packed_bbox_vs_poly.svg"
OUT_PACK_SVG_PAGE2 = SCRIPTS_DIR / "packed_page2.svg"
OUT_PACK_BBOX_SVG_PAGE2 = SCRIPTS_DIR / "packed_bbox_page2.svg"
OUT_PACK_BBOX_VS_POLY_SVG_PAGE2 = SCRIPTS_DIR / "packed_bbox_vs_poly_page2.svg"
OUT_PACK_MISSING_LOG = SCRIPTS_DIR / "packed_missing.txt"
OUT_PACK_RASTER_LOG = SCRIPTS_DIR / "packed_raster_log.txt"
OUT_ZONES_JSON = SCRIPTS_DIR / "zones_cache.json"
OUT_COLOR_PNG = SCRIPTS_DIR / "color.png"
OUT_OVERLAP_PNG = SCRIPTS_DIR / "overlap.png"
OUT_ZONE_PNG = SCRIPTS_DIR / "zone.png"
OUT_EXTENT_PNG = SCRIPTS_DIR / "extent.png"
OUT_ZONE_SVG = SCRIPTS_DIR / "zone.svg"
OUT_ZONE_OUTLINE_SVG = SCRIPTS_DIR / "zone_outline.svg"
OUT_REGION_SVG = SCRIPTS_DIR / "region.svg"
OUT_DEBUG_TRI_OUT_SVG = SCRIPTS_DIR / "debug_tri_out.svg"
OUT_DEBUG_TRI_SMALL_SVG = SCRIPTS_DIR / "debug_tri_small.svg"
OUT_DEBUG_POLY_RAW_SVG = SCRIPTS_DIR / "debug_poly_raw.svg"
OUT_DEBUG_POLY_FINAL_SVG = SCRIPTS_DIR / "debug_poly_final.svg"
OUT_PACK_LABELS_JSON = SCRIPTS_DIR / "packed_labels.json"

SNAP = 0.01
NEIGHBOR_EPS = 0.5
MIN_AREA = 1.0
SMALL_ZONE_AREA = 200.0
SMALL_ZONE_BBOX = 20.0
DRAW_SCALE = 0.5
LINE_THICKNESS = 1
FONT_SCALE = 0.12
TARGET_ZONES = 120
GRID_X = 20
GRID_Y = 10
DASH_LEN = 6
GAP_LEN = 4
STROKE_COLOR = (160, 160, 160)
WHITE_FALLBACK = (221, 221, 221)
EDGE_EPS = 0.0
MIDPOINT_EPS = 0.2
LINE_EXTEND = float(os.getenv("LINE_EXTEND", "0"))
INTERSECT_SNAP = float(os.getenv("INTERSECT_SNAP", "0.1"))
PADDING = 5
PACK_MARGIN_X = 0
PACK_MARGIN_Y = 0
PACK_GRID_STEP = 1
PACK_ANGLE_STEP = 90
PACK_MODE = os.getenv("PACK_MODE", "fast")
USE_ZONE_CACHE = False
LABEL_FONT_SCALE = 0.64
LABEL_OFFSET = 10.0
PACK_LABEL_SCALE = 2.4
PACK_BLEED = 3


def _safe_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except Exception:
        return default
    return val if math.isfinite(val) else default


def _safe_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except Exception:
        return default
    if not math.isfinite(val):
        return default
    try:
        return int(val)
    except Exception:
        return default


def _apply_pack_env() -> None:
    global DRAW_SCALE, PADDING, PACK_MARGIN_X, PACK_MARGIN_Y, PACK_BLEED, PACK_GRID_STEP, PACK_ANGLE_STEP, PACK_MODE
    global SMALL_ZONE_AREA, SMALL_ZONE_BBOX
    if "DRAW_SCALE" in os.environ:
        DRAW_SCALE = _safe_float_env("DRAW_SCALE", DRAW_SCALE)
    if "PACK_PADDING" in os.environ:
        PADDING = _safe_float_env("PACK_PADDING", PADDING)
    if "PACK_MARGIN_X" in os.environ:
        PACK_MARGIN_X = _safe_int_env("PACK_MARGIN_X", PACK_MARGIN_X)
    if "PACK_MARGIN_Y" in os.environ:
        PACK_MARGIN_Y = _safe_int_env("PACK_MARGIN_Y", PACK_MARGIN_Y)
    if "PACK_BLEED" in os.environ:
        PACK_BLEED = _safe_int_env("PACK_BLEED", PACK_BLEED)
    if "PACK_GRID_STEP" in os.environ:
        PACK_GRID_STEP = _safe_float_env("PACK_GRID_STEP", PACK_GRID_STEP)
    if "PACK_ANGLE_STEP" in os.environ:
        PACK_ANGLE_STEP = _safe_float_env("PACK_ANGLE_STEP", PACK_ANGLE_STEP)
    if "PACK_MODE" in os.environ:
        PACK_MODE = str(os.environ["PACK_MODE"]).strip().lower()
    if "SMALL_ZONE_AREA" in os.environ:
        SMALL_ZONE_AREA = _safe_float_env("SMALL_ZONE_AREA", SMALL_ZONE_AREA)
    if "SMALL_ZONE_BBOX" in os.environ:
        SMALL_ZONE_BBOX = _safe_float_env("SMALL_ZONE_BBOX", SMALL_ZONE_BBOX)
