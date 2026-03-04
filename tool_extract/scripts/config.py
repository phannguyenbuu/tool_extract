from __future__ import annotations

import os
from pathlib import Path

SVG_PATH = Path("convoi.svg")
OUT_LOG = Path("regions_log.txt")
OUT_PNG = Path("regions_log.png")
OUT_ZONES_LOG = Path("zones_log.txt")
OUT_ZONES_PNG = Path("zones_log.png")
OUT_PACK_LOG = Path("packed_log.txt")
OUT_PACK_PNG = Path("packed.png")
OUT_PACK_OUTLINE_PNG = Path("packed_outline.png")
OUT_PACK_SVG = Path("packed.svg")
OUT_PACK_BBOX_SVG = Path("packed_bbox.svg")
OUT_PACK_BBOX_VS_POLY_SVG = Path("packed_bbox_vs_poly.svg")
OUT_PACK_SVG_PAGE2 = Path("packed_page2.svg")
OUT_PACK_BBOX_SVG_PAGE2 = Path("packed_bbox_page2.svg")
OUT_PACK_BBOX_VS_POLY_SVG_PAGE2 = Path("packed_bbox_vs_poly_page2.svg")
OUT_PACK_MISSING_LOG = Path("packed_missing.txt")
OUT_PACK_RASTER_LOG = Path("packed_raster_log.txt")
OUT_ZONES_JSON = Path("zones_cache.json")
OUT_COLOR_PNG = Path("color.png")
OUT_OVERLAP_PNG = Path("overlap.png")
OUT_ZONE_PNG = Path("zone.png")
OUT_EXTENT_PNG = Path("extent.png")
OUT_ZONE_SVG = Path("zone.svg")
OUT_ZONE_OUTLINE_SVG = Path("zone_outline.svg")
OUT_REGION_SVG = Path("region.svg")
OUT_DEBUG_TRI_OUT_SVG = Path("debug_tri_out.svg")
OUT_DEBUG_TRI_SMALL_SVG = Path("debug_tri_small.svg")
OUT_DEBUG_POLY_RAW_SVG = Path("debug_poly_raw.svg")
OUT_DEBUG_POLY_FINAL_SVG = Path("debug_poly_final.svg")
OUT_PACK_LABELS_JSON = Path("packed_labels.json")

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
PADDING = 0
PACK_MARGIN_X = 0
PACK_MARGIN_Y = 0
PACK_GRID_STEP = 1
PACK_ANGLE_STEP = 90
PACK_MODE = os.getenv("PACK_MODE", "fast")
USE_ZONE_CACHE = False
LABEL_FONT_SCALE = 0.64
LABEL_OFFSET = 10.0
PACK_LABEL_SCALE = 2.4
PACK_BLEED = 3.5


def _apply_pack_env() -> None:
    global DRAW_SCALE, PADDING, PACK_MARGIN_X, PACK_MARGIN_Y, PACK_BLEED, PACK_GRID_STEP, PACK_ANGLE_STEP, PACK_MODE
    global SMALL_ZONE_AREA, SMALL_ZONE_BBOX
    if "DRAW_SCALE" in os.environ:
        DRAW_SCALE = float(os.environ["DRAW_SCALE"])
    if "PACK_PADDING" in os.environ:
        PADDING = float(os.environ["PACK_PADDING"])
    if "PACK_MARGIN_X" in os.environ:
        PACK_MARGIN_X = int(float(os.environ["PACK_MARGIN_X"]))
    if "PACK_MARGIN_Y" in os.environ:
        PACK_MARGIN_Y = int(float(os.environ["PACK_MARGIN_Y"]))
    if "PACK_BLEED" in os.environ:
        PACK_BLEED = int(float(os.environ["PACK_BLEED"]))
    if "PACK_GRID_STEP" in os.environ:
        PACK_GRID_STEP = float(os.environ["PACK_GRID_STEP"])
    if "PACK_ANGLE_STEP" in os.environ:
        PACK_ANGLE_STEP = float(os.environ["PACK_ANGLE_STEP"])
    if "PACK_MODE" in os.environ:
        PACK_MODE = str(os.environ["PACK_MODE"]).strip().lower()
    if "SMALL_ZONE_AREA" in os.environ:
        SMALL_ZONE_AREA = float(os.environ["SMALL_ZONE_AREA"])
    if "SMALL_ZONE_BBOX" in os.environ:
        SMALL_ZONE_BBOX = float(os.environ["SMALL_ZONE_BBOX"])
