from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize_full, triangulate, unary_union

from . import config
from . import svg_utils

GRAPH_TOL = 1.0


@dataclass
class RegionInfo:
    idx: int
    area: float
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]


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

    for idx in range(len(seg_list)):
        start = seg_list[idx][0]
        seg_list[idx] = sorted(seg_list[idx], key=lambda p: _dist(p, start))

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


def _iter_geoms(geom) -> Iterable:
    if geom is None:
        return []
    if hasattr(geom, "geoms"):
        return geom.geoms
    return [geom]


def build_regions_from_svg(
    svg_path: Path, snap_override: float | None = None
) -> Tuple[List[RegionInfo], List[List[Tuple[float, float]]], Tuple[int, int], Dict[str, float]]:
    root = ET.parse(svg_path).getroot()
    canvas = svg_utils._get_canvas_size(root, 1.0)

    lines: List[LineString] = []
    for gtype, pts in svg_utils._iter_geometry(root):
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

    w, h = canvas
    extent_img = np.full((int(h * config.DRAW_SCALE), int(w * config.DRAW_SCALE), 3), 255, dtype=np.uint8)
    for ln in lines:
        coords = list(ln.coords)
        if len(coords) < 2:
            continue
        pts_scaled = np.array([[x * config.DRAW_SCALE, y * config.DRAW_SCALE] for x, y in coords], dtype=np.int32)
        cv2.polylines(extent_img, [pts_scaled], False, (0, 0, 0), 1, cv2.LINE_AA)

    extended_lines = []
    for ln in lines:
        coords = list(ln.coords)
        if len(coords) < 2:
            continue
        a = (coords[0][0], coords[0][1])
        b = (coords[-1][0], coords[-1][1])
        a, b = _extend_line(a, b, config.LINE_EXTEND)
        extended_lines.append(LineString([a, b]))

    merged = unary_union(extended_lines)
    poly_pts, border_pts = polygonize_full(merged)[:2]

    raw_poly_pts: List[List[Tuple[float, float]]] = []
    for geom in _iter_geoms(poly_pts):
        if geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            coords = list(geom.exterior.coords)
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) >= 3:
                raw_poly_pts.append([(float(x), float(y)) for x, y in coords])
        elif geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                coords = list(g.exterior.coords)
                if len(coords) > 1 and coords[0] == coords[-1]:
                    coords = coords[:-1]
                if len(coords) >= 3:
                    raw_poly_pts.append([(float(x), float(y)) for x, y in coords])

    polys: List[List[Tuple[float, float]]] = []
    tri_out_pts: List[List[Tuple[float, float]]] = []
    tri_small_pts: List[List[Tuple[float, float]]] = []
    tri_total = 0
    tri_kept = 0
    tri_removed_small = 0
    tri_removed_outside = 0
    for coords in raw_poly_pts:
        if len(coords) > 3:
            try:
                tris = triangulate(Polygon(coords))
            except Exception:
                tris = []
            for tri in tris:
                tri_total += 1
                if tri.is_empty or tri.area < config.MIN_AREA:
                    tri_removed_small += 1
                    tcoords = list(tri.exterior.coords)
                    if len(tcoords) > 1 and tcoords[0] == tcoords[-1]:
                        tcoords = tcoords[:-1]
                    if len(tcoords) >= 3:
                        tri_small_pts.append([(float(x), float(y)) for x, y in tcoords])
                    continue
                if not tri.within(Polygon(coords)):
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

    svg_utils._write_svg_paths_fill_stroke(config.OUT_DEBUG_POLY_RAW_SVG, canvas[0], canvas[1], raw_poly_pts, "#4cc9f0", "#1b243d")
    svg_utils._write_svg_paths_fill_stroke(config.OUT_DEBUG_POLY_FINAL_SVG, canvas[0], canvas[1], polys, "#a8ffb0", "#1b243d")
    svg_utils._write_svg_paths_fill(config.OUT_DEBUG_TRI_OUT_SVG, canvas[0], canvas[1], tri_out_pts, "#ff2d55")
    svg_utils._write_svg_paths_fill(config.OUT_DEBUG_TRI_SMALL_SVG, canvas[0], canvas[1], tri_small_pts, "#ffb020")

    regions: List[RegionInfo] = []
    for idx, pts in enumerate(polys):
        poly = Polygon(pts)
        if poly.is_empty or poly.area < config.MIN_AREA:
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
    img = np.full((int(h * config.DRAW_SCALE), int(w * config.DRAW_SCALE), 3), 255, dtype=np.uint8)
    for pts in polys:
        if len(pts) < 2:
            continue
        pts_scaled = np.array([[p[0] * config.DRAW_SCALE, p[1] * config.DRAW_SCALE] for p in pts], dtype=np.int32)
        cv2.polylines(img, [pts_scaled], True, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(config.OUT_PNG), img)


def compute_region_colors(polys: List[List[Tuple[float, float]]], canvas: Tuple[int, int]) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
    root = ET.parse(config.SVG_PATH).getroot()
    img_path, _, _ = svg_utils._find_embedded_image(root)
    img = svg_utils._read_image_any(img_path)
    w, h = canvas
    w_px = int(w * config.DRAW_SCALE)
    h_px = int(h * config.DRAW_SCALE)
    resized = cv2.resize(img, (w_px, h_px), interpolation=cv2.INTER_AREA)

    colors: List[Tuple[int, int, int]] = []
    for pts in polys:
        poly = Polygon(pts)
        if poly.is_empty:
            colors.append(config.WHITE_FALLBACK)
            continue
        cx, cy = poly.centroid.x, poly.centroid.y
        ix_int = int(round(cx * config.DRAW_SCALE))
        iy_int = int(round(cy * config.DRAW_SCALE))
        if 0 <= ix_int < w_px and 0 <= iy_int < h_px:
            b, g, r = resized[iy_int, ix_int]
            color = (int(b), int(g), int(r))
        else:
            color = (255, 255, 255)
        if color[0] >= 245 and color[1] >= 245 and color[2] >= 245:
            color = config.WHITE_FALLBACK
        colors.append(color)

    return colors, resized


def render_color_regions(polys: List[List[Tuple[float, float]]], canvas: Tuple[int, int]) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
    w, h = canvas
    w_px = int(w * config.DRAW_SCALE)
    h_px = int(h * config.DRAW_SCALE)
    out = np.full((h_px, w_px, 3), 255, dtype=np.uint8)

    colors, resized = compute_region_colors(polys, canvas)
    for pts, color in zip(polys, colors):
        pts_scaled = np.array(
            [[p[0] * config.DRAW_SCALE, p[1] * config.DRAW_SCALE] for p in pts],
            dtype=np.int32,
        )
        cv2.fillPoly(out, [pts_scaled], color)

    cv2.imwrite(str(config.OUT_COLOR_PNG), out)

    overlap = resized.copy()
    for pts in polys:
        pts_scaled = np.array(
            [[p[0] * config.DRAW_SCALE, p[1] * config.DRAW_SCALE] for p in pts],
            dtype=np.int32,
        )
        cv2.polylines(overlap, [pts_scaled], True, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(config.OUT_OVERLAP_PNG), overlap)
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
            step = config.DASH_LEN if draw else config.GAP_LEN
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
    img = np.full((int(h * config.DRAW_SCALE), int(w * config.DRAW_SCALE), 3), 255, dtype=np.uint8)
    for rid, pts in enumerate(polys):
        if len(pts) < 2:
            continue
        pts_scaled = np.array([[p[0] * config.DRAW_SCALE, p[1] * config.DRAW_SCALE] for p in pts], dtype=np.int32)
        color = colors[rid]
        cv2.fillPoly(img, [pts_scaled], color)
        cv2.polylines(img, [pts_scaled], True, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(config.OUT_ZONES_PNG), img)
