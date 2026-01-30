from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, DefaultDict
import json
import os
import re
from urllib.parse import unquote
import unicodedata

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon, Point
from shapely.affinity import rotate as _srotate, translate as _stranslate
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from shapely.ops import polygonize_full, unary_union, triangulate
from collections import defaultdict
try:
    from shapely.validation import make_valid
except Exception:  # pragma: no cover - fallback for older shapely
    def make_valid(geom):
        return geom.buffer(0)
from rectpack import newPacker
from PIL import Image

SVG_PATH = Path("convoi.svg")
OUT_LOG = Path("regions_log.txt")
OUT_PNG = Path("regions_log.png")
OUT_ZONES_LOG = Path("zones_log.txt")
OUT_ZONES_PNG = Path("zones_log.png")
OUT_PACK_LOG = Path("packed_log.txt")
OUT_PACK_PNG = Path("packed.png")
OUT_PACK_OUTLINE_PNG = Path("packed_outline.png")
OUT_PACK_SVG = Path("packed.svg")
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

SNAP = 0.01
NEIGHBOR_EPS = 0.5
MIN_AREA = 1.0
DRAW_SCALE = 2.0
LINE_THICKNESS = 1
FONT_SCALE = 0.12
TARGET_ZONES = 99
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
PADDING = 4.0
PACK_MARGIN_X = 30
PACK_MARGIN_Y = 30
PACK_GRID_STEP = float(os.getenv("PACK_GRID_STEP", "5.0"))
PACK_ANGLE_STEP = float(os.getenv("PACK_ANGLE_STEP", "5.0"))
PACK_MODE = os.getenv("PACK_MODE", "fast")
USE_ZONE_CACHE = False
LABEL_FONT_SCALE = 0.64
LABEL_OFFSET = 10.0
PACK_LABEL_SCALE = 2.4
PACK_BLEED = 10


@dataclass
class RegionInfo:
    idx: int
    area: float
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]


def _parse_points(points_str: str) -> List[Tuple[float, float]]:
    if not points_str:
        return []
    tokens: List[str] = []
    for part in points_str.replace("\n", " ").replace("\t", " ").split():
        if "," in part:
            tokens.extend(part.split(","))
        else:
            tokens.append(part)
    pts: List[Tuple[float, float]] = []
    if len(tokens) % 2 == 0:
        for i in range(0, len(tokens), 2):
            try:
                x = float(tokens[i])
                y = float(tokens[i + 1])
                pts.append((x, y))
            except ValueError:
                continue
        return pts
    for part in points_str.strip().split():
        if "," not in part:
            continue
        x_str, y_str = part.split(",", 1)
        try:
            pts.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    return pts


def _iter_geometry(root: ET.Element) -> Iterable[Tuple[str, List[Tuple[float, float]]]]:
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "polyline":
            pts = _parse_points(el.attrib.get("points", ""))
            if len(pts) >= 2:
                yield ("polyline", pts)
        elif tag == "polygon":
            pts = _parse_points(el.attrib.get("points", ""))
            if len(pts) >= 3:
                yield ("polygon", pts)
        elif tag == "line":
            try:
                x1 = float(el.attrib.get("x1", "0"))
                y1 = float(el.attrib.get("y1", "0"))
                x2 = float(el.attrib.get("x2", "0"))
                y2 = float(el.attrib.get("y2", "0"))
                yield ("polyline", [(x1, y1), (x2, y2)])
            except ValueError:
                continue


def _get_canvas_size(root: ET.Element, scale: float) -> Tuple[int, int]:
    vb = root.attrib.get("viewBox")
    if vb:
        parts = vb.replace(",", " ").split()
        if len(parts) == 4:
            _, _, w, h = parts
            return (int(ceil(float(w) * scale)), int(ceil(float(h) * scale)))
    w_attr = root.attrib.get("width")
    h_attr = root.attrib.get("height")
    if w_attr and h_attr:
        try:
            return (int(ceil(float(w_attr) * scale)), int(ceil(float(h_attr) * scale)))
        except ValueError:
            pass
    raise RuntimeError("Cannot determine SVG canvas size")


def _parse_transform(transform: str) -> List[Tuple[str, Tuple[float, float]]]:
    ops: List[Tuple[str, Tuple[float, float]]] = []
    if not transform:
        return ops
    for name, args in re.findall(r"(translate|scale)\s*\(([^)]*)\)", transform):
        nums = [float(v) for v in re.split(r"[ ,]+", args.strip()) if v]
        if name == "translate":
            tx = nums[0] if len(nums) > 0 else 0.0
            ty = nums[1] if len(nums) > 1 else 0.0
            ops.append(("translate", (tx, ty)))
        elif name == "scale":
            sx = nums[0] if len(nums) > 0 else 1.0
            sy = nums[1] if len(nums) > 1 else sx
            ops.append(("scale", (sx, sy)))
    return ops


def _ops_to_matrix(ops: List[Tuple[str, Tuple[float, float]]]) -> np.ndarray:
    m = np.eye(3, dtype=np.float64)
    for name, params in ops:
        if name == "translate":
            tx, ty = params
            t = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
            m = t @ m
        elif name == "scale":
            sx, sy = params
            s = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
            m = s @ m
    return m


def _invert_transform_point(x: float, y: float, m: np.ndarray) -> Tuple[float, float]:
    inv = np.linalg.inv(m)
    v = np.array([x, y, 1.0], dtype=np.float64)
    out = inv @ v
    return float(out[0]), float(out[1])


def _normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


def _find_embedded_image(root: ET.Element) -> Tuple[Path, np.ndarray, Tuple[float, float]]:
    href_key = "{http://www.w3.org/1999/xlink}href"
    svg_dir = SVG_PATH.parent
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag != "image":
            continue
        href = el.attrib.get(href_key) or el.attrib.get("href")
        if not href:
            continue
        href = unquote(href)
        img_path = Path(href)
        if not img_path.is_absolute():
            img_path = (svg_dir / img_path).resolve()
        transform = el.attrib.get("transform", "")
        ops = []
        x_attr = float(el.attrib.get("x", "0"))
        y_attr = float(el.attrib.get("y", "0"))
        if x_attr != 0.0 or y_attr != 0.0:
            ops.append(("translate", (x_attr, y_attr)))
        ops.extend(_parse_transform(transform))
        # if simple translate then scale, build direct matrix = T * S
        if len(ops) == 2 and ops[0][0] == "translate" and ops[1][0] == "scale":
            tx, ty = ops[0][1]
            sx, sy = ops[1][1]
            m = np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=np.float64)
        else:
            m = _ops_to_matrix(ops)
        w = float(el.attrib.get("width", "0"))
        h = float(el.attrib.get("height", "0"))
        if img_path.exists():
            return img_path, m, (w, h)

        # fallback: search by basename within svg directory (accent-insensitive)
        basename = img_path.name
        target = _normalize_name(basename)
        candidates: List[Path] = []
        for p in svg_dir.rglob("*"):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
                continue
            if _normalize_name(p.name) == target:
                candidates.append(p)
        if candidates:
            return candidates[0], m, (w, h)
    raise RuntimeError("No embedded <image> found in SVG")


def _read_image_any(path: Path) -> np.ndarray:
    # PIL handles unicode paths more reliably on Windows
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im)
        # convert RGB -> BGR for OpenCV style
        return arr[:, :, ::-1].copy()


GRAPH_TOL = 1.0


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _extend_line(a: Tuple[float, float], b: Tuple[float, float], amt: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0 or amt <= 0:
        return a, b
    ux = dx / length
    uy = dy / length
    return (a[0] - ux * amt, a[1] - uy * amt), (b[0] + ux * amt, b[1] + uy * amt)


def _seg_intersect(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    tole: float,
    can_extents: List[bool],
) -> Tuple[bool, bool, Tuple[float, float] | None, List[int]]:
    dx12 = p1[0] - p0[0]
    dy12 = p1[1] - p0[1]
    dx34 = p3[0] - p2[0]
    dy34 = p3[1] - p2[1]

    delta = (dy12 * dx34 - dx12 * dy34)
    if delta == 0:
        return False, False, None, []

    t1 = ((p0[0] - p2[0]) * dy34 + (p2[1] - p0[1]) * dx34) / delta
    t2 = ((p2[0] - p0[0]) * dy12 + (p0[1] - p2[1]) * dx12) / -delta

    ix = p0[0] + dx12 * t1
    iy = p0[1] + dy12 * t1
    inter = (ix, iy)

    b1 = 0 <= t1 <= 1
    b2 = 0 <= t2 <= 1

    ar = [
        t1 < 0 and can_extents[0] and _dist(p0, inter) < tole,
        t1 > 1 and can_extents[1] and _dist(p1, inter) < tole,
        t2 < 0 and can_extents[2] and _dist(p2, inter) < tole,
        t2 > 1 and can_extents[3] and _dist(p3, inter) < tole,
    ]
    duplicate_indexes = [i for i, v in enumerate(ar) if v]

    seg_intersect = b1 and b2
    half_intersect = False
    if not seg_intersect:
        half_intersect = (ar[0] and ar[2]) or (ar[0] and ar[3]) or (ar[1] and ar[2]) or (ar[1] and ar[3])
        if not half_intersect:
            seg_intersect = (b1 and ar[2]) or (b1 and ar[3]) or (b2 and ar[0]) or (b2 and ar[1])

    return seg_intersect, half_intersect, inter, duplicate_indexes


def _graph_compute(pls: List[List[Tuple[float, float]]]) -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    seg_list: List[List[Tuple[float, float]]] = []
    ls_spline_index: List[int] = []
    ls_spline_ext: List[List[bool]] = []

    # AddVertList
    for spline_idx, ls in enumerate(pls):
        if len(ls) < 2:
            continue
        closed = _dist(ls[0], ls[-1]) < GRAPH_TOL
        for i in range(len(ls)):
            nex = 0 if (i == len(ls) - 1 and closed) else (i + 1 if i < len(ls) - 1 else -1)
            if nex == -1:
                continue
            seg_list.append([ls[i], ls[nex]])
            ls_spline_index.append(spline_idx)
            ar = [False, False]
            if len(ls) == 2:
                ar = [True, True]
            elif not closed:
                if i == 0:
                    ar[0] = True
                elif i == len(ls) - 2:
                    ar[1] = True
            ls_spline_ext.append(ar)

    # collectSegments
    seg_pairs: List[Tuple[int, int, Tuple[float, float], List[int]]] = []
    for i in range(len(seg_list) - 1):
        intersects_in: List[Tuple[float, float]] = []
        tmp_seg: List[Tuple[int, int, Tuple[float, float], List[int]]] = []
        for j in range(i + 1, len(seg_list)):
            if ls_spline_index[i] == ls_spline_index[j]:
                continue
            can_extents = [ls_spline_ext[i][0], ls_spline_ext[i][1], ls_spline_ext[j][0], ls_spline_ext[j][1]]
            seg_intersect, half_intersect, inter, dup = _seg_intersect(
                seg_list[i][0], seg_list[i][1], seg_list[j][0], seg_list[j][1], GRAPH_TOL, can_extents
            )
            if inter is None:
                continue
            if seg_intersect:
                seg_pairs.append((i, j, inter, dup))
                intersects_in.append(inter)
            elif half_intersect:
                tmp_seg.append((i, j, inter, dup))
        for itm in tmp_seg:
            inter = itm[2]
            if inter is None:
                continue
            if not any(_dist(pt, inter) < GRAPH_TOL for pt in intersects_in):
                seg_pairs.append(itm)

    # divide segments
    for i, j, inter, dup in seg_pairs:
        if dup:
            for idx in dup:
                a = i if idx < 2 else j
                b = 0 if idx in (0, 2) else 1
                seg_list[a][b] = inter
        if not (0 in dup and 1 in dup):
            seg_list[i].append(inter)
        if not (2 in dup and 3 in dup):
            seg_list[j].append(inter)

    # sort points on each segment
    for idx in range(len(seg_list)):
        start = seg_list[idx][0]
        seg_list[idx] = sorted(seg_list[idx], key=lambda p: _dist(p, start))

    # collect verts
    verts: List[Tuple[float, float]] = []
    adj: List[List[int]] = []

    def _add_vert(pt: Tuple[float, float]) -> int:
        for vi, v in enumerate(verts):
            if _dist(v, pt) < GRAPH_TOL:
                return vi
        verts.append(pt)
        adj.append([])
        return len(verts) - 1

    def _add_edge(v: int, w: int) -> None:
        if v == w:
            return
        if w not in adj[v]:
            adj[v].append(w)
        if v not in adj[w]:
            adj[w].append(v)

    for seg in seg_list:
        idxs = [_add_vert(pt) for pt in seg]
        for a, b in zip(idxs, idxs[1:]):
            if a != b:
                _add_edge(a, b)

    # collectAdj: remove degree-1 chains
    changed = True
    while changed:
        changed = False
        for i in range(len(adj)):
            if len(adj[i]) == 1:
                j = adj[i][0]
                if i in adj[j]:
                    adj[j].remove(i)
                adj[i] = []
                changed = True

    # sort and compute angles
    angles: List[List[float]] = [[] for _ in range(len(adj))]
    for i in range(len(adj)):
        if adj[i]:
            adj[i].sort()
            angs = []
            for j in adj[i]:
                px, py = verts[i]
                qx, qy = verts[j]
                dx = px - qx
                dy = py - qy
                angs.append(np.degrees(np.arctan2(dy, dx)))
            angles[i] = angs

    # detect cycles
    adjs = [list(a) for a in adj]
    angs = [list(a) for a in angles]
    cycles: List[List[int]] = []

    for i in range(len(adjs)):
        if len(adjs[i]) == 0:
            continue
        while len(adjs[i]) > 0:
            pos_array = [i]
            w = i
            j = adjs[i][0]
            if j != i:
                pos_array.append(j)
            nulla = angs[w][0] - 180.0
            if nulla < 0:
                nulla += 360.0
            adjs[w].pop(0)
            angs[w].pop(0)

            while j != i and len(adjs[j]) > 0:
                min_val = 360.0
                min_idx = -1
                for k_idx, cand in enumerate(adjs[j]):
                    if cand == w:
                        continue
                    x = angs[j][k_idx] - nulla
                    if x < 0:
                        x += 360.0
                    if x >= 360.0:
                        x -= 360.0
                    if x < min_val:
                        min_val = x
                        min_idx = k_idx
                if min_idx == -1:
                    break
                w = j
                j = adjs[w][min_idx]
                nulla = angs[w][min_idx] - 180.0
                if nulla < 0:
                    nulla += 360.0
                if j != i:
                    pos_array.append(j)
                adjs[w].pop(min_idx)
                angs[w].pop(min_idx)

            if len(pos_array) > 2:
                cycles.append(pos_array)

    def _poly_area(idxs: List[int]) -> float:
        pts = [verts[v] for v in idxs]
        area = 0.0
        for a, b in zip(pts, pts[1:] + [pts[0]]):
            area += a[0] * b[1] - b[0] * a[1]
        return abs(area) * 0.5

    cycles = sorted(cycles, key=_poly_area)

    # find outlines (largest in each intersecting group)
    cycle_sets = [set(c) for c in cycles]
    used = [False] * len(cycles)
    border_idxs: List[int] = []
    for i in range(len(cycles)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(cycles)):
                if used[j]:
                    continue
                if any(cycle_sets[j].intersection(cycle_sets[k]) for k in group):
                    used[j] = True
                    group.append(j)
                    changed = True
        group = sorted(group, key=lambda idx: _poly_area(cycles[idx]))
        border_idxs.append(group[-1])

    result_pts: List[List[Tuple[float, float]]] = []
    result_borders: List[List[Tuple[float, float]]] = []
    for i, cyc in enumerate(cycles):
        pts = [verts[v] for v in cyc]
        if i in border_idxs:
            result_borders.append(pts)
        else:
            result_pts.append(pts)

    def _area_pts(pts: List[Tuple[float, float]]) -> float:
        area = 0.0
        for a, b in zip(pts, pts[1:] + [pts[0]]):
            area += a[0] * b[1] - b[0] * a[1]
        return abs(area) * 0.5

    result_pts = sorted(result_pts, key=lambda pts: -_area_pts(pts))
    result_borders = sorted(result_borders, key=lambda pts: -_area_pts(pts))
    return result_pts, result_borders


def build_regions_from_svg(
    svg_path: Path, snap_override: float | None = None
) -> Tuple[List[RegionInfo], List[List[Tuple[float, float]]], Tuple[int, int], Dict[str, float]]:
    root = ET.parse(svg_path).getroot()
    canvas = _get_canvas_size(root, 1.0)

    lines: List[LineString] = []
    for gtype, pts in _iter_geometry(root):
        if gtype == "polyline":
            for a, b in zip(pts, pts[1:]):
                if a != b:
                    lines.append(LineString([a, b]))
        elif gtype == "polygon":
            for a, b in zip(pts, pts[1:] + [pts[0]]):
                if a != b:
                    lines.append(LineString([a, b]))

    if not lines:
        raise RuntimeError("No line/polyline/polygon geometry found")
    debug: Dict[str, float] = {
        "lines_in": float(len(lines)),
    }

    # log extended lines for inspection
    w, h = canvas
    extent_img = np.full((int(h * DRAW_SCALE), int(w * DRAW_SCALE), 3), 255, dtype=np.uint8)
    for ln in lines:
        coords = list(ln.coords)
        if len(coords) < 2:
            continue
        pts_scaled = np.array([[x * DRAW_SCALE, y * DRAW_SCALE] for x, y in coords], dtype=np.int32)
        cv2.polylines(extent_img, [pts_scaled], False, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT_EXTENT_PNG), extent_img)

    # node/split lines at intersections before polygonize
    noded = unary_union(lines)
    noded_lines: List[LineString] = []
    if noded.geom_type == "LineString":
        noded_lines = [noded]
    else:
        for g in getattr(noded, "geoms", []):
            if g.geom_type == "LineString":
                noded_lines.append(g)
            elif g.geom_type == "MultiLineString":
                noded_lines.extend(list(g.geoms))
    debug["lines_noded"] = float(len(noded_lines))

    # snap after intersections
    snap = INTERSECT_SNAP if snap_override is None else snap_override
    snapped_lines: List[LineString] = []
    for ln in noded_lines:
        coords = list(ln.coords)
        snapped = [(round(x / snap) * snap, round(y / snap) * snap) for x, y in coords]
        # drop zero-length segments
        if len(snapped) >= 2 and snapped[0] != snapped[-1]:
            snapped_lines.append(LineString(snapped))
    debug["lines_snapped"] = float(len(snapped_lines))

    polygons, _, _, _ = polygonize_full(snapped_lines)

    polys: List[List[Tuple[float, float]]] = []
    regions: List[RegionInfo] = []

    polys_all: List[Polygon] = []
    poly_geoms = list(polygons.geoms) if hasattr(polygons, "geoms") else list(polygons)
    raw_poly_pts: List[List[Tuple[float, float]]] = []
    debug["polygons_raw"] = float(len(poly_geoms))
    removed_small = 0
    for poly in poly_geoms:
        if poly.is_empty:
            continue
        coords = list(poly.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) >= 3:
            raw_poly_pts.append([(float(x), float(y)) for x, y in coords])
        if poly.area <= 0 or poly.area < MIN_AREA:
            removed_small += 1
            continue
        polys_all.append(poly)
    debug["polygons_kept_pre_largest"] = float(len(polys_all))
    debug["polygons_removed_small"] = float(removed_small)

    if not polys_all:
        raise RuntimeError("No regions from polygonize")

    # keep largest polygon as region (do not drop)
    largest = max(polys_all, key=lambda p: p.area)
    debug["polygons_removed_largest"] = 0.0
    tri_total = 0
    tri_kept = 0
    tri_removed_small = 0
    tri_removed_outside = 0
    tri_out_pts: List[List[Tuple[float, float]]] = []
    tri_small_pts: List[List[Tuple[float, float]]] = []
    for poly in polys_all:
        if poly.is_empty or poly.area <= 0 or poly.area < MIN_AREA:
            continue
        # triangulate polygons with >3 vertices
        coords = list(poly.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) > 3:
            try:
                tris = triangulate(poly)
            except Exception:
                tris = []
            for tri in tris:
                tri_total += 1
                if tri.is_empty or tri.area < MIN_AREA:
                    tri_removed_small += 1
                    tcoords = list(tri.exterior.coords)
                    if len(tcoords) > 1 and tcoords[0] == tcoords[-1]:
                        tcoords = tcoords[:-1]
                    if len(tcoords) >= 3:
                        tri_small_pts.append([(float(x), float(y)) for x, y in tcoords])
                    continue
                if not tri.within(poly):
                    tri_removed_outside += 1
                    tcoords = list(tri.exterior.coords)
                    if len(tcoords) > 1 and tcoords[0] == tcoords[-1]:
                        tcoords = tcoords[:-1]
                    if len(tcoords) >= 3:
                        tri_out_pts.append([(float(x), float(y)) for x, y in tcoords])
                    continue
                tcoords = list(tri.exterior.coords)
                if len(tcoords) > 1 and tcoords[0] == tcoords[-1]:
                    tcoords = tcoords[:-1]
                if len(tcoords) == 3:
                    polys.append([(float(x), float(y)) for x, y in tcoords])
                    tri_kept += 1
        else:
            polys.append([(float(x), float(y)) for x, y in coords])
    debug["tri_total"] = float(tri_total)
    debug["tri_kept"] = float(tri_kept)
    debug["tri_removed_small"] = float(tri_removed_small)
    debug["tri_removed_outside"] = float(tri_removed_outside)
    debug["polygons_final"] = float(len(polys))

    _write_svg_paths_fill_stroke(OUT_DEBUG_POLY_RAW_SVG, canvas[0], canvas[1], raw_poly_pts, "#4cc9f0", "#1b243d")
    _write_svg_paths_fill_stroke(OUT_DEBUG_POLY_FINAL_SVG, canvas[0], canvas[1], polys, "#a8ffb0", "#1b243d")
    _write_svg_paths_fill(OUT_DEBUG_TRI_OUT_SVG, canvas[0], canvas[1], tri_out_pts, "#ff2d55")
    _write_svg_paths_fill(OUT_DEBUG_TRI_SMALL_SVG, canvas[0], canvas[1], tri_small_pts, "#ffb020")

    # safety: ensure all polygons are triangles
    if any(len(pts) > 3 for pts in polys):
        tri_polys: List[List[Tuple[float, float]]] = []
        for pts in polys:
            if len(pts) <= 3:
                tri_polys.append(pts)
                continue
            poly = Polygon(pts)
            if poly.is_empty or poly.area < MIN_AREA:
                continue
            try:
                tris = triangulate(poly)
            except Exception:
                tris = []
            for tri in tris:
                if tri.is_empty or tri.area < MIN_AREA:
                    continue
                if not tri.within(poly):
                    continue
                tcoords = list(tri.exterior.coords)
                if len(tcoords) > 1 and tcoords[0] == tcoords[-1]:
                    tcoords = tcoords[:-1]
                if len(tcoords) == 3:
                    tri_polys.append([(float(x), float(y)) for x, y in tcoords])
        polys = tri_polys

    for idx, pts in enumerate(polys):
        poly = Polygon(pts)
        if poly.is_empty or poly.area < MIN_AREA:
            continue
        minx, miny, maxx, maxy = poly.bounds
        regions.append(
            RegionInfo(
                idx=idx,
                area=float(poly.area),
                bbox=(float(minx), float(miny), float(maxx), float(maxy)),
                centroid=(float(poly.centroid.x), float(poly.centroid.y)),
            )
        )

    return regions, polys, canvas, debug


def write_log(regions: List[RegionInfo], out_path: Path) -> None:
    lines = [f"count={len(regions)}"]
    for r in regions:
        lines.append(
            f"id={r.idx} area={r.area:.2f} bbox={r.bbox} centroid=({r.centroid[0]:.2f},{r.centroid[1]:.2f})"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_png(polys: List[List[Tuple[float, float]]], regions: List[RegionInfo], canvas: Tuple[int, int]) -> None:
    w, h = canvas
    img = np.full((int(h * DRAW_SCALE), int(w * DRAW_SCALE), 3), 255, dtype=np.uint8)
    for pts in polys:
        if len(pts) < 2:
            continue
        pts_scaled = np.array([[p[0] * DRAW_SCALE, p[1] * DRAW_SCALE] for p in pts], dtype=np.int32)
        cv2.polylines(img, [pts_scaled], True, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT_PNG), img)


def build_zones(polys: List[List[Tuple[float, float]]], target: int) -> List[int]:
    # simple grid-based assignment as fallback
    if not polys:
        return []
    minx = min(p[0] for poly in polys for p in poly)
    miny = min(p[1] for poly in polys for p in poly)
    maxx = max(p[0] for poly in polys for p in poly)
    maxy = max(p[1] for poly in polys for p in poly)
    cell_w = (maxx - minx) / GRID_X
    cell_h = (maxy - miny) / GRID_Y

    zones: List[List[int]] = []
    zone_id = [-1] * len(polys)
    poly_objs = [Polygon(p) for p in polys]

    for gy in range(GRID_Y):
        for gx in range(GRID_X):
            x0 = minx + gx * cell_w
            y0 = miny + gy * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            cell = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            members = []
            for rid, poly in enumerate(poly_objs):
                if zone_id[rid] != -1:
                    continue
                if poly.intersects(cell):
                    members.append(rid)
            if members:
                zid = len(zones)
                zones.append(members)
                for rid in members:
                    zone_id[rid] = zid

    # assign any remaining to nearest zone
    remaining = [rid for rid, zid in enumerate(zone_id) if zid == -1]
    for rid in remaining:
        c = poly_objs[rid].centroid
        best = 0
        best_d = 1e18
        for zid, members in enumerate(zones):
            if not members:
                continue
            cm = unary_union([poly_objs[m] for m in members]).centroid
            d = (cm.x - c.x) ** 2 + (cm.y - c.y) ** 2
            if d < best_d:
                best_d = d
                best = zid
        zones[best].append(rid)
        zone_id[rid] = best

    # merge smallest zones until target reached
    while len(zones) > target and zones:
        smallest = min(range(len(zones)), key=lambda i: sum(poly_objs[r].area for r in zones[i]))
        if len(zones) == 1:
            break
        # merge into nearest zone by centroid distance
        c_small = unary_union([poly_objs[r] for r in zones[smallest]]).centroid
        best = None
        best_d = 1e18
        for zid in range(len(zones)):
            if zid == smallest or not zones[zid]:
                continue
            c_other = unary_union([poly_objs[r] for r in zones[zid]]).centroid
            d = (c_other.x - c_small.x) ** 2 + (c_other.y - c_small.y) ** 2
            if d < best_d:
                best_d = d
                best = zid
        if best is None:
            break
        for rid in zones[smallest]:
            zones[best].append(rid)
            zone_id[rid] = best
        zones.pop(smallest)
        # reindex zone_id
        for i, members in enumerate(zones):
            for rid in members:
                zone_id[rid] = i

    return zone_id


def compute_scene(svg_path: Path, snap: float) -> Dict:
    _apply_pack_env()
    regions, polys, canvas, debug = build_regions_from_svg(svg_path, snap_override=snap)
    zone_id = build_zones(polys, TARGET_ZONES)
    zone_boundaries = build_zone_boundaries(polys, zone_id)
    zone_geoms = build_zone_geoms(polys, zone_id)
    zone_polys, zone_order, zone_poly_debug = build_zone_polys(polys, zone_id)
    placements, _, rot_info = pack_regions(
        zone_polys,
        canvas,
        allow_rotate=True,
        angle_step=PACK_ANGLE_STEP,
        grid_step=PACK_GRID_STEP,
    )

    zone_labels = {}
    for zid, geom in zone_geoms.items():
        lx, ly = _label_pos_for_zone(geom)
        zone_labels[str(zid)] = {"x": lx, "y": ly, "label": zid}

    region_labels = {}
    for rid, pts in enumerate(polys):
        poly = Polygon(pts)
        if poly.is_empty:
            continue
        region_labels[str(rid)] = {
            "x": float(poly.centroid.x),
            "y": float(poly.centroid.y),
            "label": rid,
            "zone": zone_id[rid] if rid < len(zone_id) else -1,
        }

    packed_polys: List[List[Tuple[float, float]]] = []
    packed_colors: List[str] = []
    region_colors: List[str] = []
    packed_labels: Dict[str, Dict[str, float]] = {}
    packed_zone_polys: List[List[Tuple[float, float]]] = []
    zone_shift: Dict[int, Tuple[float, float]] = {}
    zone_rot: Dict[int, float] = {}
    zone_center: Dict[int, Tuple[float, float]] = {}

    for idx, zid in enumerate(zone_order):
        if idx >= len(placements):
            continue
        dx, dy, _, _, _ = placements[idx]
        if dx < 0 and dy < 0:
            continue
        zone_shift[zid] = (dx, dy)
        info = rot_info[idx] if idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
        zone_rot[zid] = float(info.get("angle", 0.0))
        zone_center[zid] = (float(info.get("cx", 0.0)), float(info.get("cy", 0.0)))

    colors, _ = compute_region_colors(polys, canvas)
    region_colors = [f"#{r:02x}{g:02x}{b:02x}" for (b, g, r) in colors]
    for idx, zid in enumerate(zone_order):
        if idx >= len(zone_polys):
            continue
        if zid not in zone_shift:
            continue
        dx, dy = zone_shift[zid]
        pts = zone_polys[idx]
        ang = zone_rot.get(zid, 0.0)
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        packed = [(p[0] + dx, p[1] + dy) for p in _rotate_pts(pts, ang, cx, cy)]
        packed_zone_polys.append(packed)

    for rid, pts in enumerate(polys):
        if rid >= len(zone_id):
            continue
        zid = zone_id[rid]
        if zid not in zone_shift:
            continue
        dx, dy = zone_shift[zid]
        ang = zone_rot.get(zid, 0.0)
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        packed = [(p[0] + dx, p[1] + dy) for p in _rotate_pts(pts, ang, cx, cy)]
        packed_polys.append(packed)
        packed_colors.append(region_colors[rid])

    for zid in zone_geoms:
        if zid not in zone_shift:
            continue
        dx, dy = zone_shift[zid]
        lx, ly = _label_pos_for_zone(zone_geoms[zid])
        ang = zone_rot.get(zid, 0.0)
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        lx, ly = _rotate_pts([(lx, ly)], ang, cx, cy)[0]
        packed_labels[str(zid)] = {"x": lx + dx, "y": ly + dy, "label": zid}

    debug["zones_total"] = float(max(zone_id) + 1) if zone_id else 0.0
    debug["packed_placed"] = float(len(zone_shift))
    debug["zones_empty"] = zone_poly_debug.get("empty", [])
    debug["zones_convex_hull"] = zone_poly_debug.get("convex_hull", [])

    return {
        "canvas": {"w": canvas[0], "h": canvas[1]},
        "regions": polys,
        "zone_boundaries": zone_boundaries,
        "zone_id": zone_id,
        "zone_labels": zone_labels,
        "region_labels": region_labels,
        "packed_polys": packed_polys,
        "packed_colors": packed_colors,
        "packed_labels": packed_labels,
        "packed_zone_polys": packed_zone_polys,
        "region_colors": region_colors,
        "debug": debug,
        "snap": snap,
    }


def save_zones_cache(zone_id: List[int], polys: List[List[Tuple[float, float]]], out_path: Path) -> None:
    zones: Dict[int, List[int]] = {}
    for rid, zid in enumerate(zone_id):
        zones.setdefault(zid, []).append(rid)
    data = {
        "zones": {str(k): v for k, v in zones.items()},
        "polys": polys,
        "svg_mtime": os.path.getmtime(SVG_PATH),
    }
    out_path.write_text(json.dumps(data), encoding="utf-8")


def load_zones_cache(path: Path) -> Tuple[List[List[Tuple[float, float]]], List[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    polys = [list(map(lambda p: (p[0], p[1]), pts)) for pts in data["polys"]]
    zone_id = [-1] * len(polys)
    for k, members in data["zones"].items():
        zid = int(k)
        for rid in members:
            zone_id[rid] = zid
    return polys, zone_id



def compute_region_colors(polys: List[List[Tuple[float, float]]], canvas: Tuple[int, int]) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
    root = ET.parse(SVG_PATH).getroot()
    img_path, _, _ = _find_embedded_image(root)
    img = _read_image_any(img_path)
    w, h = canvas
    w_px = int(w * DRAW_SCALE)
    h_px = int(h * DRAW_SCALE)
    resized = cv2.resize(img, (w_px, h_px), interpolation=cv2.INTER_AREA)

    colors: List[Tuple[int, int, int]] = []
    for pts in polys:
        poly = Polygon(pts)
        if poly.is_empty:
            colors.append(WHITE_FALLBACK)
            continue
        cx, cy = poly.centroid.x, poly.centroid.y
        ix_int = int(round(cx * DRAW_SCALE))
        iy_int = int(round(cy * DRAW_SCALE))
        if 0 <= ix_int < w_px and 0 <= iy_int < h_px:
            b, g, r = resized[iy_int, ix_int]
            color = (int(b), int(g), int(r))
        else:
            color = (255, 255, 255)
        if color[0] >= 245 and color[1] >= 245 and color[2] >= 245:
            color = WHITE_FALLBACK
        colors.append(color)

    return colors, resized


def render_color_regions(polys: List[List[Tuple[float, float]]], canvas: Tuple[int, int]) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
    w, h = canvas
    w_px = int(w * DRAW_SCALE)
    h_px = int(h * DRAW_SCALE)
    out = np.full((h_px, w_px, 3), 255, dtype=np.uint8)

    colors, resized = compute_region_colors(polys, canvas)
    for pts, color in zip(polys, colors):
        pts_scaled = np.array(
            [[p[0] * DRAW_SCALE, p[1] * DRAW_SCALE] for p in pts],
            dtype=np.int32,
        )
        cv2.fillPoly(out, [pts_scaled], color)

    cv2.imwrite(str(OUT_COLOR_PNG), out)

    # overlap view: image scaled to canvas + region strokes
    overlap = resized.copy()
    for pts in polys:
        pts_scaled = np.array(
            [[p[0] * DRAW_SCALE, p[1] * DRAW_SCALE] for p in pts],
            dtype=np.int32,
        )
        cv2.polylines(overlap, [pts_scaled], True, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT_OVERLAP_PNG), overlap)
    return colors, resized


def _draw_dashed_polyline(img: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int]) -> None:
    if len(pts) < 2:
        return
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        seg_len = (dx * dx + dy * dy) ** 0.5
        if seg_len == 0:
            continue
        ux = dx / seg_len
        uy = dy / seg_len
        dist = 0.0
        draw = True
        while dist < seg_len:
            step = DASH_LEN if draw else GAP_LEN
            dist2 = min(dist + step, seg_len)
            if draw:
                sx = x1 + ux * dist
                sy = y1 + uy * dist
                ex = x1 + ux * dist2
                ey = y1 + uy * dist2
                cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)), color, 1, cv2.LINE_AA)
            dist = dist2
            draw = not draw


def write_zones_png(
    polys: List[List[Tuple[float, float]]],
    zone_id: List[int],
    canvas: Tuple[int, int],
    colors: List[Tuple[int, int, int]],
) -> None:
    w, h = canvas
    img = np.full((int(h * DRAW_SCALE), int(w * DRAW_SCALE), 3), 255, dtype=np.uint8)

    if zone_id:
        for rid, pts in enumerate(polys):
            pts_scaled = np.array([[p[0] * DRAW_SCALE, p[1] * DRAW_SCALE] for p in pts], dtype=np.int32)
            color = colors[rid]
            cv2.fillPoly(img, [pts_scaled], color)
            closed = np.vstack([pts_scaled, pts_scaled[0]])
            _draw_dashed_polyline(img, closed, STROKE_COLOR)

    cv2.imwrite(str(OUT_ZONES_PNG), img)


def build_bleed(group_mask: np.ndarray, canvas_fill: np.ndarray, border_thick: int) -> Tuple[np.ndarray, np.ndarray]:
    border = int(border_thick // 2)
    h, w = group_mask.shape
    bleed_color_img = canvas_fill.copy()

    current_mask = (group_mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(border):
        dilated = cv2.dilate(current_mask, kernel, iterations=1)
        ring = (dilated == 1) & (current_mask == 0)
        if not ring.any():
            break

        src = bleed_color_img.copy()
        yy, xx = np.where(ring)
        for y, x in zip(yy, xx):
            y0, y1 = max(0, y - 1), min(h, y + 2)
            x0, x1 = max(0, x - 1), min(w, x + 2)
            neighbours_mask = current_mask[y0:y1, x0:x1]
            neighbours_img = src[y0:y1, x0:x1]
            ys2, xs2 = np.where(neighbours_mask == 1)
            if len(ys2) == 0:
                continue
            bleed_color_img[y, x] = neighbours_img[ys2[0], xs2[0]]

        current_mask = dilated

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1))
    dilated_final = cv2.dilate(group_mask, kernel2)
    bleed_mask = ((dilated_final > 0) & (group_mask == 0)).astype(np.uint8) * 255

    return bleed_mask, bleed_color_img



def write_zones_log(zone_id: List[int], out_path: Path, zone_labels: Dict[int, int] | None = None) -> None:
    if not zone_id:
        out_path.write_text("total_zones=0\n", encoding="utf-8")
        return
    n_zones = max(zone_id) + 1
    zones: List[List[int]] = [[] for _ in range(n_zones)]
    for rid, zid in enumerate(zone_id):
        if zid >= 0:
            zones[zid].append(rid)
    lines = [f"total_zones={n_zones}"]
    if zone_labels:
        inv = {v: k for k, v in zone_labels.items()}
        for shuffle_idx in range(1, n_zones + 1):
            zid = inv.get(shuffle_idx)
            if zid is None:
                continue
            members = zones[zid]
            lines.append(
                f"zone_shuffle_index={shuffle_idx} zone_id={zid} size={len(members)} regions={members}"
            )
    else:
        for zid, members in enumerate(zones):
            lines.append(f"zone_id={zid} size={len(members)} regions={members}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_zone_polys(
    polys: List[List[Tuple[float, float]]], zone_id: List[int]
) -> Tuple[List[List[Tuple[float, float]]], List[int], Dict[str, List[int]]]:
    zones: Dict[int, List[Polygon]] = {}
    for rid, zid in enumerate(zone_id):
        zones.setdefault(zid, []).append(Polygon(polys[rid]))
    zone_polys: List[List[Tuple[float, float]]] = []
    zone_order = sorted(zones.keys())
    debug: Dict[str, List[int]] = {"empty": [], "convex_hull": []}
    for zid in zone_order:
        merged = unary_union(zones[zid])
        if merged.is_empty:
            # fallback: tiny square to keep zone count stable
            zone_polys.append([(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)])
            debug["empty"].append(zid)
            continue
        if merged.geom_type != "Polygon":
            merged = merged.convex_hull
            debug["convex_hull"].append(zid)
        zone_polys.append(list(merged.exterior.coords))
    return zone_polys, zone_order, debug


def _rotate_pts(pts: List[Tuple[float, float]], angle_deg: float, cx: float, cy: float) -> List[Tuple[float, float]]:
    if angle_deg == 0:
        return [(float(x), float(y)) for x, y in pts]
    ang = np.deg2rad(angle_deg)
    c = float(np.cos(ang))
    s = float(np.sin(ang))
    out = []
    for x, y in pts:
        rx = x - cx
        ry = y - cy
        out.append((cx + rx * c - ry * s, cy + rx * s + ry * c))
    return out


def pack_regions(
    polys: List[List[Tuple[float, float]]],
    canvas: Tuple[int, int],
    allow_rotate: bool = True,
    angle_step: float = 5.0,
    grid_step: float = 5.0,
) -> Tuple[List[Tuple[int, int, int, int, bool]], List[int], List[Dict[str, float]]]:
    """Polygon bin packing with rotation sampling."""
    w, h = canvas
    pad = float(PADDING)
    x_min = PACK_MARGIN_X
    y_min = PACK_MARGIN_Y
    x_max = w - PACK_MARGIN_X
    y_max = h - PACK_MARGIN_Y

    if PACK_MODE == "fast":
        bboxes: List[Tuple[int, float, float, int, int]] = []
        rot_info: List[Dict[str, float]] = []
        for i, pts in enumerate(polys):
            poly = Polygon(pts)
            if poly.is_empty:
                x0 = y0 = 0.0
                x1 = y1 = 1.0
                angle = 0.0
                cx = cy = 0.0
            else:
                cx, cy = float(poly.centroid.x), float(poly.centroid.y)
                angle = 0.0
                best_area = 1e18
                best_bounds = None
                if allow_rotate:
                    ang = 0.0
                    while ang < 180.0:
                        rpts = _rotate_pts(pts, ang, cx, cy)
                        xs = [p[0] for p in rpts]
                        ys = [p[1] for p in rpts]
                        x0t, y0t, x1t, y1t = min(xs), min(ys), max(xs), max(ys)
                        area = (x1t - x0t) * (y1t - y0t)
                        if area < best_area:
                            best_area = area
                            best_bounds = (x0t, y0t, x1t, y1t)
                            angle = ang
                        ang += angle_step
                if best_bounds is None:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x0t, y0t, x1t, y1t = min(xs), min(ys), max(xs), max(ys)
                    best_bounds = (x0t, y0t, x1t, y1t)
                x0, y0, x1, y1 = best_bounds
            bw = int(ceil((x1 - x0) + pad * 2))
            bh = int(ceil((y1 - y0) + pad * 2))
            bboxes.append((i, x0, y0, bw, bh))
            rot_info.append({"angle": angle, "cx": cx, "cy": cy, "minx": x0, "miny": y0})

        packer = newPacker(rotation=False)
        packer.add_bin(w - PACK_MARGIN_X * 2, h - PACK_MARGIN_Y * 2)
        for idx, _, _, bw, bh in bboxes:
            packer.add_rect(bw, bh, rid=idx)
        packer.pack()

        placements: List[Tuple[int, int, int, int, bool]] = [(-1, -1, 0, 0, False)] * len(polys)
        order: List[int] = []
        rects = list(packer.rect_list())
        if not rects:
            return placements, order, rot_info

        min_x = min(r[1] for r in rects)
        min_y = min(r[2] for r in rects)
        max_x = max(r[1] + r[3] for r in rects)
        max_y = max(r[2] + r[4] for r in rects)
        content_w = max_x - min_x
        content_h = max_y - min_y
        offset_x = PACK_MARGIN_X + max(0, (w - PACK_MARGIN_X * 2 - content_w) // 2)
        offset_y = PACK_MARGIN_Y + max(0, (h - PACK_MARGIN_Y * 2 - content_h) // 2)

        for _, x, y, pw, ph, rid in rects:
            orig = bboxes[rid]
            x0 = orig[1]
            y0 = orig[2]
            placements[rid] = (x + offset_x + pad - int(x0), y + offset_y + pad - int(y0), pw, ph, False)
            order.append(rid)

        return placements, order, rot_info

    items: List[Tuple[int, Polygon]] = []
    for i, pts in enumerate(polys):
        poly = Polygon(pts)
        if poly.is_empty:
            poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly = make_valid(poly)
        items.append((i, poly))

    items.sort(key=lambda t: t[1].area, reverse=True)

    placements: List[Tuple[int, int, int, int, bool]] = [(-1, -1, 0, 0, False)] * len(polys)
    rot_info: List[Dict[str, float]] = [
        {"angle": 0.0, "cx": 0.0, "cy": 0.0, "minx": 0.0, "miny": 0.0} for _ in polys
    ]
    placed_order: List[int] = []
    placed_polys: List[Polygon] = []

    for rid, poly in items:
        cx, cy = float(poly.centroid.x), float(poly.centroid.y)
        candidates = []
        if allow_rotate:
            ang = 0.0
            while ang < 180.0:
                rpoly = _srotate(poly, ang, origin=(cx, cy), use_radians=False)
                minx, miny, maxx, maxy = rpoly.bounds
                candidates.append((ang, rpoly, minx, miny, maxx, maxy))
                ang += angle_step
        else:
            minx, miny, maxx, maxy = poly.bounds
            candidates.append((0.0, poly, minx, miny, maxx, maxy))

        candidates.sort(key=lambda c: (c[4] - c[2]) * (c[5] - c[3]))

        placed = False
        for ang, rpoly, minx, miny, maxx, maxy in candidates:
            bw = (maxx - minx) + pad * 2
            bh = (maxy - miny) + pad * 2
            if bw > (x_max - x_min) or bh > (y_max - y_min):
                continue
            y = y_min
            while y + bh <= y_max and not placed:
                x = x_min
                while x + bw <= x_max and not placed:
                    dx = x - minx + pad
                    dy = y - miny + pad
                    tpoly = _stranslate(rpoly, xoff=dx, yoff=dy)
                    tpoly_buf = tpoly.buffer(pad)
                    collision = False
                    for p in placed_polys:
                        if tpoly_buf.intersects(p) and not tpoly_buf.touches(p):
                            collision = True
                            break
                    if not collision:
                        placements[rid] = (int(dx), int(dy), int(bw), int(bh), False)
                        rot_info[rid] = {"angle": float(ang), "cx": cx, "cy": cy, "minx": minx, "miny": miny}
                        placed_order.append(rid)
                        placed_polys.append(tpoly_buf)
                        placed = True
                        break
                    x += grid_step
                y += grid_step
            if placed:
                break

    return placements, placed_order, rot_info


def write_pack_log(
    polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[int, int, int, int, bool]],
    out_path: Path,
    canvas: Tuple[int, int],
) -> None:
    w, h = canvas
    visible = 0
    overflow = 0
    lines = ["packed_regions"]
    for rid, (dx, dy, bw, bh, rot) in enumerate(placements):
        x0 = dx
        y0 = dy
        x1 = dx + bw
        y1 = dy + bh
        in_view = not (x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h)
        out_view = x0 < 0 or y0 < 0 or x1 > w or y1 > h
        if in_view:
            visible += 1
        if out_view:
            overflow += 1
        lines.append(
            f"id={rid} dx={dx} dy={dy} bbox_w={bw} bbox_h={bh} rot={int(rot)} visible={int(in_view)} overflow={int(out_view)}"
        )
    lines.insert(0, f"visible_pieces={visible}")
    lines.insert(1, f"overflow_pieces={overflow}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_zone_geoms(polys: List[List[Tuple[float, float]]], zone_id: List[int]) -> Dict[int, BaseGeometry]:
    zones: Dict[int, List[Polygon]] = {}
    for rid, zid in enumerate(zone_id):
        zones.setdefault(zid, []).append(Polygon(polys[rid]))
    zone_geoms: Dict[int, BaseGeometry] = {}
    for zid, parts in zones.items():
        merged = unary_union(parts)
        if merged.is_empty:
            continue
        # clean geometry to avoid self-intersections while preserving thin features
        try:
            merged = make_valid(merged)
        except Exception:
            pass
        if merged.is_empty:
            continue
        zone_geoms[zid] = merged
    return zone_geoms


def _label_pos_for_zone(geom: BaseGeometry) -> Tuple[float, float]:
    # concave -> representative_point inside; convex -> place outside along nearest boundary
    if geom.is_empty:
        return (0.0, 0.0)
    hull = geom.convex_hull
    concave = (hull.area - geom.area) > 1e-3
    if concave:
        p = geom.representative_point()
        return (float(p.x), float(p.y))
    c = geom.centroid
    bnd = geom.boundary
    # nearest boundary point to centroid
    nearest = bnd.interpolate(bnd.project(c))
    vx = nearest.x - c.x
    vy = nearest.y - c.y
    norm = (vx * vx + vy * vy) ** 0.5
    if norm == 0:
        return (float(c.x), float(c.y))
    ux = vx / norm
    uy = vy / norm
    return (float(nearest.x + ux * LABEL_OFFSET), float(nearest.y + uy * LABEL_OFFSET))


def _label_pos_outside(geom: BaseGeometry, offset: float) -> Tuple[float, float]:
    if geom.is_empty:
        return (0.0, 0.0)
    c = geom.centroid
    bnd = geom.boundary
    nearest = bnd.interpolate(bnd.project(c))
    vx = nearest.x - c.x
    vy = nearest.y - c.y
    norm = (vx * vx + vy * vy) ** 0.5
    if norm == 0:
        return (float(c.x), float(c.y))
    ux = vx / norm
    uy = vy / norm
    return (float(nearest.x + ux * offset), float(nearest.y + uy * offset))


def _snap_key(pt: Tuple[float, float], snap: float) -> Tuple[float, float]:
    # no snapping/rounding; keep vector precision
    return (pt[0], pt[1])


def build_zone_boundaries(
    polys: List[List[Tuple[float, float]]],
    zone_id: List[int],
    snap: float = EDGE_EPS,
) -> Dict[int, List[List[Tuple[float, float]]]]:
    zones: Dict[int, List[List[Tuple[float, float]]]] = {}
    for rid, zid in enumerate(zone_id):
        zones.setdefault(zid, []).append(polys[rid])

    zone_boundaries: Dict[int, List[List[Tuple[float, float]]]] = {}
    for zid, zpolys in zones.items():
        edge_counts: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
        point_sum: DefaultDict[Tuple[int, int], List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

        for pts in zpolys:
            if len(pts) < 2:
                continue
            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(pts)]
                k1 = _snap_key(p1, snap)
                k2 = _snap_key(p2, snap)
                if k1 == k2:
                    continue
                point_sum[k1][0] += p1[0]
                point_sum[k1][1] += p1[1]
                point_sum[k1][2] += 1.0
                point_sum[k2][0] += p2[0]
                point_sum[k2][1] += p2[1]
                point_sum[k2][2] += 1.0
                ek = (k1, k2) if k1 < k2 else (k2, k1)
                edge_counts[ek] = edge_counts.get(ek, 0) + 1

        boundary_edges = [ek for ek, c in edge_counts.items() if c == 1]
        if not boundary_edges:
            zone_boundaries[zid] = []
            continue

        point_map: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for k, (sx, sy, cnt) in point_sum.items():
            if cnt > 0:
                point_map[k] = (sx / cnt, sy / cnt)

        kept_edges = boundary_edges

        adj: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
        for idx, (k1, k2) in enumerate(kept_edges):
            adj[k1].append(idx)
            adj[k2].append(idx)

        # keep all remaining edges (no pruning / cycle filtering)
        adj = defaultdict(list)
        for idx, (k1, k2) in enumerate(kept_edges):
            adj[k1].append(idx)
            adj[k2].append(idx)

        used = [False] * len(kept_edges)
        polylines: List[List[Tuple[float, float]]] = []

        def _next_edge(cur_key: Tuple[int, int], prev_key: Tuple[int, int] | None) -> int | None:
            candidates = [ei for ei in adj[cur_key] if not used[ei]]
            if not candidates:
                return None
            if prev_key is None:
                return candidates[0]
            # pick edge with smallest turn angle
            best_ei = None
            best_dot = -1e9
            pcur = point_map[cur_key]
            pprev = point_map[prev_key]
            vx = pcur[0] - pprev[0]
            vy = pcur[1] - pprev[1]
            vlen = (vx * vx + vy * vy) ** 0.5
            if vlen == 0:
                return candidates[0]
            for ei in candidates:
                a, b = kept_edges[ei]
                nxt = b if a == cur_key else a
                pnxt = point_map[nxt]
                wx = pnxt[0] - pcur[0]
                wy = pnxt[1] - pcur[1]
                wlen = (wx * wx + wy * wy) ** 0.5
                if wlen == 0:
                    continue
                dot = (vx * wx + vy * wy) / (vlen * wlen)
                if dot > best_dot:
                    best_dot = dot
                    best_ei = ei
            return best_ei if best_ei is not None else candidates[0]

        for i, (k1, k2) in enumerate(kept_edges):
            if used[i]:
                continue
            # DFS walk to build outline polyline
            used[i] = True
            path_keys = [k1, k2]

            # extend forward
            while True:
                cur = path_keys[-1]
                prev = path_keys[-2] if len(path_keys) >= 2 else None
                next_edge = _next_edge(cur, prev)
                if next_edge is None:
                    break
                used[next_edge] = True
                a, b = kept_edges[next_edge]
                nxt = b if a == cur else a
                if nxt == path_keys[-1]:
                    break
                path_keys.append(nxt)
                if nxt == path_keys[0]:
                    break

            # extend backward
            while True:
                cur = path_keys[0]
                prev = path_keys[1] if len(path_keys) >= 2 else None
                next_edge = _next_edge(cur, prev)
                if next_edge is None:
                    break
                used[next_edge] = True
                a, b = kept_edges[next_edge]
                nxt = b if a == cur else a
                if nxt == path_keys[0]:
                    break
                path_keys.insert(0, nxt)
                if nxt == path_keys[-1]:
                    break

            path_pts = [point_map[k] for k in path_keys if k in point_map]
            if len(path_pts) >= 2:
                polylines.append(path_pts)

        zone_boundaries[zid] = polylines

    return zone_boundaries


def write_zone_outline_png(
    zone_geoms: Dict[int, BaseGeometry],
    zone_labels: Dict[int, int],
    canvas: Tuple[int, int],
    zone_boundaries: Dict[int, List[List[Tuple[float, float]]]],
) -> None:
    w, h = canvas
    zone_scale = DRAW_SCALE * 2
    img = np.full((int(h * zone_scale), int(w * zone_scale), 3), 255, dtype=np.uint8)

    for zid, paths in zone_boundaries.items():
        for pts in paths:
            pts_scaled = np.array([[x * zone_scale, y * zone_scale] for x, y in pts], dtype=np.int32)
            closed = len(pts_scaled) > 2 and np.array_equal(pts_scaled[0], pts_scaled[-1])
            cv2.polylines(img, [pts_scaled], closed, (0, 0, 0), 2, cv2.LINE_AA)

    for zid, geom in zone_geoms.items():
        lx, ly = _label_pos_for_zone(geom)
        label = str(zone_labels.get(zid, zid))
        cv2.putText(
            img,
            label,
            (int(lx * zone_scale), int(ly * zone_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            LABEL_FONT_SCALE,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    img_out = cv2.resize(img, (int(w * DRAW_SCALE), int(h * DRAW_SCALE)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT_ZONE_PNG), img_out)


def _write_svg_paths(path: Path, width: int, height: int, polys: List[List[Tuple[float, float]]]) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    for pts in polys:
        if len(pts) < 2:
            continue
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in pts) + " Z"
        parts.append(f'<path d="{d}" fill="none" stroke="#000" stroke-width="1"/>')
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def _write_svg_paths_fill(path: Path, width: int, height: int, polys: List[List[Tuple[float, float]]], color: str) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    for pts in polys:
        if len(pts) < 2:
            continue
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in pts) + " Z"
        parts.append(f'<path d="{d}" fill="{color}" stroke="none"/>')
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def _write_svg_paths_fill_stroke(
    path: Path,
    width: int,
    height: int,
    polys: List[List[Tuple[float, float]]],
    fill: str,
    stroke: str,
    opacity: float = 0.6,
) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#0b0f1f"/>')
    for pts in polys:
        if len(pts) < 2:
            continue
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in pts) + " Z"
        parts.append(
            f'<path d="{d}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="0.5"/>'
        )
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def _apply_pack_env() -> None:
    global PADDING, PACK_MARGIN_X, PACK_MARGIN_Y, PACK_BLEED, PACK_GRID_STEP, PACK_ANGLE_STEP, PACK_MODE
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


def write_region_svg(polys: List[List[Tuple[float, float]]], canvas: Tuple[int, int]) -> None:
    w, h = canvas
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]
    for rid, pts in enumerate(polys):
        if len(pts) < 2:
            continue
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in pts) + " Z"
        parts.append(f'<path d="{d}" fill="#bbb" stroke="#000" stroke-width="1"/>')
        poly = Polygon(pts)
        if poly.is_empty:
            continue
        cx, cy = poly.centroid.x, poly.centroid.y
        parts.append(
            f'<text x="{cx}" y="{cy}" font-size="6" text-anchor="middle" '
            f'dominant-baseline="middle" fill="#111">{rid}</text>'
        )
    parts.append("</svg>")
    OUT_REGION_SVG.write_text("".join(parts), encoding="utf-8")


def write_zone_svg(
    polys: List[List[Tuple[float, float]]],
    zone_boundaries: Dict[int, List[List[Tuple[float, float]]]],
    canvas: Tuple[int, int],
    colors: List[Tuple[int, int, int]],
) -> None:
    w, h = canvas
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]
    parts.append('<g id="fill">')
    for rid, pts in enumerate(polys):
        if len(pts) < 2:
            continue
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in pts) + " Z"
        b, g, r = colors[rid]
        parts.append(f'<path d="{d}" fill="rgb({r},{g},{b})" stroke="none"/>')
    parts.append("</g>")
    parts.append('<g id="outline">')
    for _, lines in zone_boundaries.items():
        for pts in lines:
            if len(pts) < 2:
                continue
            d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in pts)
            parts.append(f'<path d="{d}" fill="none" stroke="#000" stroke-width="1"/>')
    parts.append("</g></svg>")
    OUT_ZONE_SVG.write_text("".join(parts), encoding="utf-8")


def write_zone_outline_svg(
    zone_boundaries: Dict[int, List[List[Tuple[float, float]]]],
    canvas: Tuple[int, int],
) -> None:
    paths: List[List[Tuple[float, float]]] = []
    for _, lines in zone_boundaries.items():
        for pts in lines:
            if len(pts) >= 2:
                paths.append(pts)
    _write_svg_paths(OUT_ZONE_OUTLINE_SVG, canvas[0], canvas[1], paths)


def write_pack_png(
    polys: List[List[Tuple[float, float]]],
    zone_id: List[int],
    zone_order: List[int],
    zone_polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[int, int, int, int, bool]],
    canvas: Tuple[int, int],
    colors: List[Tuple[int, int, int]],
    zone_geoms: Dict[int, BaseGeometry],
    zone_labels: Dict[int, int],
    region_labels: Dict[int, int],
    rot_info: List[Dict[str, float]],
) -> None:
    w, h = canvas
    hi_scale = DRAW_SCALE * 2
    img = np.full((int(h * hi_scale), int(w * hi_scale), 3), 255, dtype=np.uint8)
    zone_shift: Dict[int, Tuple[float, float]] = {}
    zone_rot: Dict[int, float] = {}
    zone_center: Dict[int, Tuple[float, float]] = {}
    for idx, zid in enumerate(zone_order):
        dx, dy, _, _, _ = placements[idx]
        zone_shift[zid] = (dx, dy)
        info = rot_info[idx] if idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
        zone_rot[zid] = float(info.get("angle", 0.0))
        zone_center[zid] = (float(info.get("cx", 0.0)), float(info.get("cy", 0.0)))

    for rid, pts in enumerate(polys):
        zid = zone_id[rid]
        if zid not in zone_shift:
            continue
        dx, dy = zone_shift[zid]
        ang = zone_rot.get(zid, 0.0)
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        rpts = _rotate_pts(pts, ang, cx, cy)
        pts_shifted = np.array(
            [[(p[0] + dx) * hi_scale, (p[1] + dy) * hi_scale] for p in rpts],
            dtype=np.int32,
        )
        color = colors[rid]
        cv2.fillPoly(img, [pts_shifted], color)

    # bleed per zone
    if PACK_BLEED > 0:
        base_img = img.copy()
        for zid, (dx, dy) in zone_shift.items():
            mask = np.zeros((int(h * hi_scale), int(w * hi_scale)), dtype=np.uint8)
            for rid, pts in enumerate(polys):
                if zone_id[rid] != zid:
                    continue
                ang = zone_rot.get(zid, 0.0)
                cx, cy = zone_center.get(zid, (0.0, 0.0))
                rpts = _rotate_pts(pts, ang, cx, cy)
                pts_shifted = np.array(
                    [[(p[0] + dx) * hi_scale, (p[1] + dy) * hi_scale] for p in rpts],
                    dtype=np.int32,
                )
                cv2.fillPoly(mask, [pts_shifted], 255)
            bleed_mask, bleed_color_img = build_bleed(mask, base_img, PACK_BLEED)
            img[bleed_mask > 0] = bleed_color_img[bleed_mask > 0]

    # draw vector outlines on top of color layer
    for rid, pts in enumerate(polys):
        zid = zone_id[rid]
        if zid not in zone_shift:
            continue
        dx, dy = zone_shift[zid]
        ang = zone_rot.get(zid, 0.0)
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        rpts = _rotate_pts(pts, ang, cx, cy)
        pts_shifted = np.array(
            [[(p[0] + dx) * hi_scale, (p[1] + dy) * hi_scale] for p in rpts],
            dtype=np.int32,
        )
        cv2.polylines(img, [pts_shifted], True, (0, 0, 0), 1, cv2.LINE_AA)

    # piece count at top-left (zone count) using same font as labels
    visible = 0
    for zid, geom in zone_geoms.items():
        if zid not in zone_shift:
            continue
        dx, dy = zone_shift[zid]
        x0, y0, x1, y1 = geom.bounds
        x0 += dx
        x1 += dx
        y0 += dy
        y1 += dy
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            continue
        visible += 1
    cv2.putText(
        img,
        f"Zones: {visible}/{len(zone_geoms)}",
        (int(20 * hi_scale / DRAW_SCALE), int(30 * hi_scale / DRAW_SCALE)),
        cv2.FONT_HERSHEY_SIMPLEX,
        PACK_LABEL_SCALE,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # draw shuffled zone index outside each zone
    for zid, geom in zone_geoms.items():
        if zid not in zone_shift:
            continue
        dx, dy = zone_shift[zid]
        lx, ly = _label_pos_outside(geom, LABEL_OFFSET)
        ang = zone_rot.get(zid, 0.0)
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        lx, ly = _rotate_pts([(lx, ly)], ang, cx, cy)[0]
        lx += dx
        ly += dy
        label = str(zone_labels.get(zid, zid))
        cv2.putText(
            img,
            label,
            (int(lx * hi_scale), int(ly * hi_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            PACK_LABEL_SCALE,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # downscale for crisp text
    img_out = cv2.resize(img, (int(w * DRAW_SCALE), int(h * DRAW_SCALE)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT_PACK_PNG), img_out)


def write_pack_svg(
    polys: List[List[Tuple[float, float]]],
    zone_id: List[int],
    zone_order: List[int],
    zone_polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[int, int, int, int, bool]],
    canvas: Tuple[int, int],
    colors: List[Tuple[int, int, int]],
    rot_info: List[Dict[str, float]],
) -> None:
    w, h = canvas
    zone_shift: Dict[int, Tuple[float, float]] = {}
    zone_rot: Dict[int, float] = {}
    zone_center: Dict[int, Tuple[float, float]] = {}
    for idx, zid in enumerate(zone_order):
        if idx >= len(placements):
            continue
        dx, dy, _, _, _ = placements[idx]
        zone_shift[zid] = (dx, dy)
        info = rot_info[idx] if idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
        zone_rot[zid] = float(info.get("angle", 0.0))
        zone_center[zid] = (float(info.get("cx", 0.0)), float(info.get("cy", 0.0)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]
    parts.append('<g id="fill">')
    for rid, pts in enumerate(polys):
        if rid >= len(zone_id):
            continue
        zid = zone_id[rid]
        if zid not in zone_shift:
            continue
        dx, dy = zone_shift[zid]
        ang = zone_rot.get(zid, 0.0)
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        moved = [(p[0] + dx, p[1] + dy) for p in _rotate_pts(pts, ang, cx, cy)]
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in moved) + " Z"
        b, g, r = colors[rid]
        parts.append(f'<path d="{d}" fill="rgb({r},{g},{b})" stroke="none"/>')
    parts.append("</g>")
    parts.append('<g id="outline">')
    for rid, pts in enumerate(polys):
        if rid >= len(zone_id):
            continue
        zid = zone_id[rid]
        if zid not in zone_shift:
            continue
        dx, dy = zone_shift[zid]
        ang = zone_rot.get(zid, 0.0)
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        moved = [(p[0] + dx, p[1] + dy) for p in _rotate_pts(pts, ang, cx, cy)]
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in moved) + " Z"
        parts.append(f'<path d="{d}" fill="none" stroke="#000" stroke-width="1"/>')
    parts.append("</g></svg>")
    OUT_PACK_SVG.write_text("".join(parts), encoding="utf-8")


def write_pack_outline_png(
    polys: List[List[Tuple[float, float]]],
    zone_id: List[int],
    zone_order: List[int],
    zone_polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[int, int, int, int, bool]],
    canvas: Tuple[int, int],
    zone_geoms: Dict[int, BaseGeometry],
    rot_info: List[Dict[str, float]],
) -> None:
    w, h = canvas
    hi_scale = DRAW_SCALE * 2
    img = np.full((int(h * hi_scale), int(w * hi_scale), 3), 255, dtype=np.uint8)
    zone_shift: Dict[int, Tuple[float, float]] = {}
    zone_rot: Dict[int, float] = {}
    zone_center: Dict[int, Tuple[float, float]] = {}
    for idx, zid in enumerate(zone_order):
        dx, dy, _, _, _ = placements[idx]
        zone_shift[zid] = (dx, dy)
        info = rot_info[idx] if idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
        zone_rot[zid] = float(info.get("angle", 0.0))
        zone_center[zid] = (float(info.get("cx", 0.0)), float(info.get("cy", 0.0)))

    for rid, pts in enumerate(polys):
        zid = zone_id[rid]
        if zid not in zone_shift:
            continue
        dx, dy = zone_shift[zid]
        ang = zone_rot.get(zid, 0.0)
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        rpts = _rotate_pts(pts, ang, cx, cy)
        pts_shifted = np.array(
            [[(p[0] + dx) * hi_scale, (p[1] + dy) * hi_scale] for p in rpts],
            dtype=np.int32,
        )
        cv2.polylines(img, [pts_shifted], True, (0, 0, 0), 1, cv2.LINE_AA)

    img_out = cv2.resize(img, (int(w * DRAW_SCALE), int(h * DRAW_SCALE)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT_PACK_OUTLINE_PNG), img_out)


def main() -> None:
    if not SVG_PATH.exists():
        raise SystemExit(f"Missing {SVG_PATH}")
    _apply_pack_env()

    svg_mtime = os.path.getmtime(SVG_PATH)
    cache_ok = False
    if USE_ZONE_CACHE and OUT_ZONES_JSON.exists():
        try:
            data = json.loads(OUT_ZONES_JSON.read_text(encoding="utf-8"))
            cache_ok = float(data.get("svg_mtime", -1)) >= svg_mtime
        except Exception:
            cache_ok = False

    if cache_ok:
        polys, zone_id = load_zones_cache(OUT_ZONES_JSON)
        base_canvas = _get_canvas_size(ET.parse(SVG_PATH).getroot(), 1.0)
        canvas = base_canvas
        regions = [RegionInfo(i, 0.0, (0, 0, 0, 0), (0.0, 0.0)) for i in range(len(polys))]
    else:
        regions, polys, canvas, _ = build_regions_from_svg(SVG_PATH)
        write_log(regions, OUT_LOG)
        write_png(polys, regions, canvas)

        zone_id = build_zones(polys, TARGET_ZONES)
        write_zones_log(zone_id, OUT_ZONES_LOG)
        save_zones_cache(zone_id, polys, OUT_ZONES_JSON)

    colors, _ = render_color_regions(polys, _get_canvas_size(ET.parse(SVG_PATH).getroot(), 1.0))
    write_zones_png(polys, zone_id, canvas, colors)

    zone_polys, zone_order, _ = build_zone_polys(polys, zone_id)
    zone_geoms = build_zone_geoms(polys, zone_id)
    # shuffle zone indices
    zone_ids = sorted(zone_geoms.keys())
    rng = np.random.default_rng(42)
    shuffled = zone_ids.copy()
    rng.shuffle(shuffled)
    zone_labels = {z: idx + 1 for idx, z in enumerate(shuffled)}
    zone_boundaries = build_zone_boundaries(polys, zone_id)
    write_zone_outline_png(zone_geoms, zone_labels, canvas, zone_boundaries)
    write_zone_svg(polys, zone_boundaries, canvas, colors)
    write_zone_outline_svg(zone_boundaries, canvas)
    write_region_svg(polys, canvas)
    write_zones_log(zone_id, OUT_ZONES_LOG, zone_labels)

    # shuffle region labels
    region_ids = list(range(len(polys)))
    rng = np.random.default_rng(43)
    rng.shuffle(region_ids)
    region_labels = {rid: idx + 1 for idx, rid in enumerate(region_ids)}
    base_canvas = _get_canvas_size(ET.parse(SVG_PATH).getroot(), 1.0)
    placements, _, rot_info = pack_regions(zone_polys, base_canvas, allow_rotate=True, angle_step=5.0)
    write_pack_log(zone_polys, placements, OUT_PACK_LOG, base_canvas)
    write_pack_png(polys, zone_id, zone_order, zone_polys, placements, base_canvas, colors, zone_geoms, zone_labels, region_labels, rot_info)
    write_pack_svg(polys, zone_id, zone_order, zone_polys, placements, base_canvas, colors, rot_info)
    write_pack_outline_png(polys, zone_id, zone_order, zone_polys, placements, base_canvas, zone_geoms, rot_info)

    total_zones = max(zone_id) + 1 if zone_id else 0
    print(
        f"Wrote {OUT_LOG}, {OUT_PNG}, {OUT_ZONES_LOG}, {OUT_ZONES_PNG}, {OUT_PACK_LOG}, {OUT_PACK_PNG} "
        f"with {len(regions)} regions and {total_zones} zones"
    )


if __name__ == "__main__":
    main()
