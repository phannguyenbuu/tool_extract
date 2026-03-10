from __future__ import annotations

import math
import random
import re
from collections import defaultdict
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.spatial import Voronoi
from shapely import intersection as shapely_intersection
from shapely.errors import GEOSException
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, polygonize_full, unary_union
from shapely.validation import make_valid

from . import config
from . import geometry
from . import svg_utils


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _parse_points_attr(raw: str) -> List[Tuple[float, float]]:
    vals: List[float] = []
    for part in raw.replace(",", " ").split():
        try:
            vals.append(float(part))
        except ValueError:
            continue
    return list(zip(vals[0::2], vals[1::2]))


def _polygon_from_pts(pts: List[Tuple[float, float]]) -> Polygon | None:
    if len(pts) < 3:
        return None
    try:
        poly = Polygon(pts)
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not isinstance(poly, Polygon) or poly.area <= 1e-6:
        return None
    return poly


def _polygon_pieces(geom) -> list[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.area > 1e-6 else []
    pieces: list[Polygon] = []
    for part in getattr(geom, "geoms", []):
        if isinstance(part, Polygon) and part.area > 1e-6:
            pieces.append(part)
    return pieces


def _coerce_polygon(geom, *, allow_largest: bool = False) -> Polygon | None:
    if geom is None:
        return None
    candidates = [geom]
    try:
        candidates.append(geom.buffer(0))
    except Exception:
        pass
    try:
        candidates.append(make_valid(geom))
    except Exception:
        pass
    for candidate in candidates:
        pieces = _polygon_pieces(candidate)
        if not pieces:
            continue
        if len(pieces) == 1:
            piece = pieces[0]
        elif allow_largest:
            piece = max(pieces, key=lambda g: g.area)
        else:
            continue
        try:
            if not piece.is_valid:
                piece = piece.buffer(0)
        except Exception:
            continue
        if isinstance(piece, Polygon) and not piece.is_empty and piece.area > 1e-6:
            return piece
    return None


def _safe_intersection_polygon(left, right, *, allow_largest: bool = False) -> Polygon | None:
    left_poly = _coerce_polygon(left, allow_largest=allow_largest)
    right_poly = _coerce_polygon(right, allow_largest=allow_largest)
    if left_poly is None or right_poly is None:
        return None
    attempts = [
        lambda a, b: a.intersection(b),
        lambda a, b: shapely_intersection(a, b, grid_size=1e-3),
        lambda a, b: make_valid(a).intersection(make_valid(b)),
        lambda a, b: shapely_intersection(make_valid(a), make_valid(b), grid_size=1e-3),
    ]
    for op in attempts:
        try:
            geom = op(left_poly, right_poly)
        except GEOSException:
            continue
        except Exception:
            continue
        poly = _coerce_polygon(geom, allow_largest=allow_largest)
        if poly is not None:
            return poly
    return None


def _select_mask_polygons(polygons: List[Polygon]) -> List[Polygon]:
    if len(polygons) <= 1:
        return polygons

    sorted_polys = sorted(polygons, key=lambda p: p.area, reverse=True)
    largest_poly = sorted_polys[0]
    areas = [p.area for p in sorted_polys]
    largest = areas[0]
    second = areas[1]
    median = areas[len(areas) // 2]

    contains_count = 0
    for poly in sorted_polys[1:]:
        rep = poly.representative_point()
        if largest_poly.contains(rep):
            contains_count += 1

    if largest > second * 8 and largest > median * 20 and contains_count >= len(sorted_polys[1:]) * 0.75:
        candidate_polys = sorted_polys[1:]
        edge_minx, edge_miny, edge_maxx, edge_maxy = largest_poly.bounds
        trimmed: List[Polygon] = []
        for poly in candidate_polys:
            minx, miny, maxx, maxy = poly.bounds
            touches_outer_ring = (
                abs(minx - edge_minx) < 0.5
                or abs(miny - edge_miny) < 0.5
                or abs(maxx - edge_maxx) < 0.5
                or abs(maxy - edge_maxy) < 0.5
            )
            if not touches_outer_ring:
                trimmed.append(poly)
        return trimmed or candidate_polys

    return polygons


def _cubic_bezier(p0, p1, p2, p3, n: int = 16):
    out = []
    for i in range(1, n + 1):
        t = i / n
        mt = 1 - t
        x = (mt**3) * p0[0] + 3 * (mt**2) * t * p1[0] + 3 * mt * (t**2) * p2[0] + (t**3) * p3[0]
        y = (mt**3) * p0[1] + 3 * (mt**2) * t * p1[1] + 3 * mt * (t**2) * p2[1] + (t**3) * p3[1]
        out.append((x, y))
    return out


def _quad_bezier(p0, p1, p2, n: int = 16):
    out = []
    for i in range(1, n + 1):
        t = i / n
        mt = 1 - t
        x = (mt**2) * p0[0] + 2 * mt * t * p1[0] + (t**2) * p2[0]
        y = (mt**2) * p0[1] + 2 * mt * t * p1[1] + (t**2) * p2[1]
        out.append((x, y))
    return out


def _angle_between(u, v):
    dot = u[0] * v[0] + u[1] * v[1]
    nu = math.hypot(u[0], u[1])
    nv = math.hypot(v[0], v[1])
    if nu == 0 or nv == 0:
        return 0.0
    c = max(-1.0, min(1.0, dot / (nu * nv)))
    ang = math.acos(c)
    cross = u[0] * v[1] - u[1] * v[0]
    return -ang if cross < 0 else ang


def _arc_to_points(x1, y1, rx, ry, phi_deg, large_arc, sweep, x2, y2, n: int = 24):
    if rx == 0 or ry == 0:
        return [(x2, y2)]
    phi = math.radians(phi_deg % 360.0)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    dx2 = (x1 - x2) / 2.0
    dy2 = (y1 - y2) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    rx = abs(rx)
    ry = abs(ry)
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1:
        s = math.sqrt(lam)
        rx *= s
        ry *= s

    sign = -1.0 if large_arc == sweep else 1.0
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = 0.0 if den == 0 else sign * math.sqrt(max(0.0, num / den))
    cxp = coef * (rx * y1p / ry)
    cyp = coef * (-ry * x1p / rx)

    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

    v1 = ((x1p - cxp) / rx, (y1p - cyp) / ry)
    v2 = ((-x1p - cxp) / rx, (-y1p - cyp) / ry)

    theta1 = _angle_between((1, 0), v1)
    delta = _angle_between(v1, v2)
    if (not sweep) and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi

    segs = max(8, int(abs(delta) / (math.pi / 12)) + 1, n)
    out = []
    for i in range(1, segs + 1):
        t = theta1 + delta * (i / segs)
        ct = math.cos(t)
        st = math.sin(t)
        x = cos_phi * rx * ct - sin_phi * ry * st + cx
        y = sin_phi * rx * ct + cos_phi * ry * st + cy
        out.append((x, y))
    return out


def _parse_path_d(d: str):
    tokens = re.findall(r"[AaCcHhLlMmQqSsTtVvZz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    i = 0
    cmd = None
    x = y = 0.0
    start = None
    pts: List[Tuple[float, float]] = []
    subpaths = []
    last_ctrl_cubic = None
    last_ctrl_quad = None

    def flush():
        nonlocal pts
        if len(pts) >= 3:
            subpaths.append(pts[:])
        pts = []

    def append_point(px, py):
        nonlocal x, y
        x, y = px, py
        pts.append((x, y))

    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"[AaCcHhLlMmQqSsTtVvZz]", tok):
            cmd = tok
            i += 1
            if cmd in "Zz":
                if pts and start and pts[-1] != start:
                    pts.append(start)
                flush()
                start = None
                last_ctrl_cubic = None
                last_ctrl_quad = None
            continue

        if cmd is None:
            i += 1
            continue

        if cmd in "Mm":
            nx = float(tok)
            ny = float(tokens[i + 1])
            i += 2
            if cmd == "m":
                nx += x
                ny += y
            if pts:
                flush()
            pts = [(nx, ny)]
            x, y = nx, ny
            start = (x, y)
            cmd = "l" if cmd == "m" else "L"
            last_ctrl_cubic = None
            last_ctrl_quad = None
        elif cmd in "Ll":
            nx = float(tok)
            ny = float(tokens[i + 1])
            i += 2
            if cmd == "l":
                nx += x
                ny += y
            append_point(nx, ny)
            last_ctrl_cubic = None
            last_ctrl_quad = None
        elif cmd in "Hh":
            nx = float(tok)
            i += 1
            nx = nx + x if cmd == "h" else nx
            append_point(nx, y)
            last_ctrl_cubic = None
            last_ctrl_quad = None
        elif cmd in "Vv":
            ny = float(tok)
            i += 1
            ny = ny + y if cmd == "v" else ny
            append_point(x, ny)
            last_ctrl_cubic = None
            last_ctrl_quad = None
        elif cmd in "Cc":
            x1 = float(tok)
            y1 = float(tokens[i + 1])
            x2 = float(tokens[i + 2])
            y2 = float(tokens[i + 3])
            x3 = float(tokens[i + 4])
            y3 = float(tokens[i + 5])
            i += 6
            if cmd == "c":
                p1 = (x + x1, y + y1)
                p2 = (x + x2, y + y2)
                p3 = (x + x3, y + y3)
            else:
                p1 = (x1, y1)
                p2 = (x2, y2)
                p3 = (x3, y3)
            for px, py in _cubic_bezier((x, y), p1, p2, p3):
                append_point(px, py)
            last_ctrl_cubic = p2
            last_ctrl_quad = None
        elif cmd in "Ss":
            x2 = float(tok)
            y2 = float(tokens[i + 1])
            x3 = float(tokens[i + 2])
            y3 = float(tokens[i + 3])
            i += 4
            if last_ctrl_cubic is None:
                p1 = (x, y)
            else:
                p1 = (2 * x - last_ctrl_cubic[0], 2 * y - last_ctrl_cubic[1])
            if cmd == "s":
                p2 = (x + x2, y + y2)
                p3 = (x + x3, y + y3)
            else:
                p2 = (x2, y2)
                p3 = (x3, y3)
            for px, py in _cubic_bezier((x, y), p1, p2, p3):
                append_point(px, py)
            last_ctrl_cubic = p2
            last_ctrl_quad = None
        elif cmd in "Qq":
            x1 = float(tok)
            y1 = float(tokens[i + 1])
            x2 = float(tokens[i + 2])
            y2 = float(tokens[i + 3])
            i += 4
            if cmd == "q":
                p1 = (x + x1, y + y1)
                p2 = (x + x2, y + y2)
            else:
                p1 = (x1, y1)
                p2 = (x2, y2)
            for px, py in _quad_bezier((x, y), p1, p2):
                append_point(px, py)
            last_ctrl_quad = p1
            last_ctrl_cubic = None
        elif cmd in "Tt":
            x2 = float(tok)
            y2 = float(tokens[i + 1])
            i += 2
            if last_ctrl_quad is None:
                p1 = (x, y)
            else:
                p1 = (2 * x - last_ctrl_quad[0], 2 * y - last_ctrl_quad[1])
            p2 = (x + x2, y + y2) if cmd == "t" else (x2, y2)
            for px, py in _quad_bezier((x, y), p1, p2):
                append_point(px, py)
            last_ctrl_quad = p1
            last_ctrl_cubic = None
        elif cmd in "Aa":
            rx = float(tok)
            ry = float(tokens[i + 1])
            phi = float(tokens[i + 2])
            large_arc = int(float(tokens[i + 3]))
            sweep = int(float(tokens[i + 4]))
            x2 = float(tokens[i + 5])
            y2 = float(tokens[i + 6])
            i += 7
            if cmd == "a":
                x2 += x
                y2 += y
            for px, py in _arc_to_points(x, y, rx, ry, phi, large_arc, sweep, x2, y2):
                append_point(px, py)
            last_ctrl_cubic = None
            last_ctrl_quad = None
        else:
            i += 1

    flush()
    return subpaths


def _parse_viewbox(root: ET.Element) -> List[float]:
    vb = root.get("viewBox")
    if vb:
        vals = [float(x) for x in vb.replace(",", " ").split()]
        if len(vals) == 4:
            return vals
    w = float((root.get("width") or "1000").replace("px", ""))
    h = float((root.get("height") or "1000").replace("px", ""))
    return [0.0, 0.0, w, h]


def _parse_source(svg_bytes: bytes):
    root = ET.fromstring(svg_bytes)
    vb = _parse_viewbox(root)
    canvas_area = max(vb[2] * vb[3], 1.0)
    vb_minx, vb_miny, vb_w, vb_h = vb
    vb_maxx = vb_minx + vb_w
    vb_maxy = vb_miny + vb_h

    raw_vertices: List[Tuple[float, float]] = []
    face_polys: List[Polygon] = []
    rect_polys: List[Polygon] = []
    line_geoms: List[LineString] = []

    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag in {"defs", "clipPath"}:
            continue
        if tag == "polygon":
            pts = _parse_points_attr(elem.get("points", ""))
            raw_vertices.extend(pts)
            poly = _polygon_from_pts(pts)
            if poly:
                face_polys.append(poly)
        elif tag == "polyline":
            pts = _parse_points_attr(elem.get("points", ""))
            raw_vertices.extend(pts)
            if len(pts) >= 2:
                line_geoms.append(LineString(pts))
            if len(pts) >= 4 and pts[0] == pts[-1]:
                poly = _polygon_from_pts(pts[:-1])
                if poly:
                    face_polys.append(poly)
        elif tag == "path":
            d = elem.get("d", "")
            for pts in _parse_path_d(d):
                raw_vertices.extend(pts)
                ring = pts[:-1] if len(pts) >= 2 and pts[0] == pts[-1] else pts
                poly = _polygon_from_pts(ring)
                if poly and poly.area < canvas_area * 0.98:
                    face_polys.append(poly)
        elif tag == "rect":
            try:
                x = float(elem.get("x", "0"))
                y = float(elem.get("y", "0"))
                w = float(elem.get("width", "0"))
                h = float(elem.get("height", "0"))
            except Exception:
                continue
            pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            raw_vertices.extend(pts)
            poly = _polygon_from_pts(pts)
            if poly:
                rect_polys.append(poly)
                if poly.area < canvas_area * 0.25:
                    face_polys.append(poly)
        elif tag == "line":
            try:
                p1 = (float(elem.get("x1", "0")), float(elem.get("y1", "0")))
                p2 = (float(elem.get("x2", "0")), float(elem.get("y2", "0")))
            except Exception:
                continue
            raw_vertices.extend([p1, p2])
            touches_canvas_edge = (
                (abs(p1[0] - vb_minx) < 0.5 and abs(p2[0] - vb_minx) < 0.5)
                or (abs(p1[1] - vb_miny) < 0.5 and abs(p2[1] - vb_miny) < 0.5)
                or (abs(p1[0] - vb_maxx) < 0.5 and abs(p2[0] - vb_maxx) < 0.5)
                or (abs(p1[1] - vb_maxy) < 0.5 and abs(p2[1] - vb_maxy) < 0.5)
            )
            if not touches_canvas_edge:
                line_geoms.append(LineString([p1, p2]))

    if not raw_vertices:
        raise ValueError("cannot parse source geometry")

    face_polys = _select_mask_polygons(face_polys)

    if not face_polys and line_geoms:
        derived: List[Polygon] = []
        try:
            for poly in polygonize(unary_union(line_geoms)):
                if poly.is_valid and poly.area > 1.0:
                    derived.append(poly)
        except Exception:
            derived = []
        face_polys = _select_mask_polygons(derived)

    if not face_polys:
        if rect_polys:
            face_polys.append(max(rect_polys, key=lambda g: g.area))
        else:
            x0, y0, w, h = vb
            fallback = _polygon_from_pts([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)])
            if fallback:
                face_polys.append(fallback)

    if not face_polys:
        raise ValueError("source svg has no closed geometry for mask")

    mask = unary_union(face_polys).buffer(0)
    if isinstance(mask, MultiPolygon):
        parts = [g.buffer(0) for g in mask.geoms if g.area > 1.0]
        if not parts:
            raise ValueError("cannot build source mask")
        mask = unary_union(parts).buffer(0)
        if isinstance(mask, MultiPolygon):
            mask = max(mask.geoms, key=lambda g: g.area).buffer(0)

    if mask.is_empty or mask.area <= 1e-6:
        raise ValueError("invalid source mask")

    uniq: List[Tuple[float, float]] = []
    seen = set()
    for x, y in raw_vertices:
        key = (round(x, 3), round(y, 3))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((float(x), float(y)))

    return vb, mask, uniq


def _random_points_in_polygon(poly: Polygon, count: int, rng: random.Random):
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    attempts = 0
    while len(pts) < count and attempts < count * 5000:
        attempts += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if poly.contains(Point(x, y)):
            pts.append((x, y))
    if len(pts) < count:
        raise RuntimeError("cannot sample enough voronoi seeds")
    return np.array(pts)


def _voronoi_finite_polygons_2d(vor: Voronoi, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0 or v1 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        centroid = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - centroid[1], vs[:, 0] - centroid[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def _representative(poly: Polygon):
    p = poly.representative_point()
    return p.x, p.y


def _poly_to_vertices(poly: Polygon):
    return [(float(x), float(y)) for x, y in list(poly.exterior.coords)[:-1]]


def _poly_to_points(poly: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in list(poly.exterior.coords)[:-1]]


def _nearest_vertex(pt, targets):
    x, y = pt
    best = None
    best_d = None
    for tx, ty in targets:
        d = (tx - x) ** 2 + (ty - y) ** 2
        if best_d is None or d < best_d:
            best_d = d
            best = (tx, ty)
    return best


def _vertex_key(pt: tuple[float, float]) -> tuple[float, float]:
    return (round(float(pt[0]), 4), round(float(pt[1]), 4))


def _build_global_snap_map(cells: list[Polygon], targets, max_dist: float) -> dict[tuple[float, float], tuple[float, float]]:
    snap_map: dict[tuple[float, float], tuple[float, float]] = {}
    for poly in cells:
        for pt in _poly_to_vertices(poly):
            key = _vertex_key(pt)
            if key in snap_map:
                continue
            q = _nearest_vertex(pt, targets)
            if q is not None and math.dist(pt, q) <= max_dist:
                snap_map[key] = (float(q[0]), float(q[1]))
    return snap_map


def _snap_polygon_with_map(poly: Polygon, mask: Polygon, snap_map: dict[tuple[float, float], tuple[float, float]]):
    src = _poly_to_vertices(poly)
    snapped = []
    moved_keys: set[tuple[float, float]] = set()
    for p in src:
        key = _vertex_key(p)
        q = snap_map.get(key)
        if q is not None:
            snapped.append(q)
            moved_keys.add(key)
        else:
            snapped.append(p)

    clean = []
    for p in snapped:
        if not clean or math.dist(clean[-1], p) > 1e-6:
            clean.append(p)
    if len(clean) >= 2 and math.dist(clean[0], clean[-1]) < 1e-6:
        clean.pop()
    if len(clean) < 3:
        return poly, moved_keys, False
    try:
        cand = Polygon(clean).buffer(0).intersection(mask).buffer(0)
        if cand.is_empty:
            return poly, moved_keys, False
        if isinstance(cand, MultiPolygon):
            cand = max(cand.geoms, key=lambda g: g.area)
        if cand.area <= 1e-6:
            return poly, moved_keys, False
        return cand, moved_keys, True
    except Exception:
        return poly, moved_keys, False


def _snap_cells_consistent(
    cells: list[Polygon],
    targets,
    mask: Polygon,
    max_dist: float = 14.0,
    max_area_delta_ratio: float = 0.28,
    boundary_area_delta_ratio: float = 0.5,
) -> list[Polygon]:
    snap_map = _build_global_snap_map(cells, targets, max_dist)
    if not snap_map:
        return cells[:]

    while True:
        changed = False
        invalid_keys: set[tuple[float, float]] = set()
        for poly in cells:
            cand, moved_keys, _ = _snap_polygon_with_map(poly, mask, snap_map)
            if not moved_keys:
                continue
            base_area = max(float(poly.area), 1e-6)
            area_delta_ratio = abs(float(cand.area) - base_area) / base_area
            touches_boundary = poly.boundary.intersection(mask.boundary).length > 1e-3
            allowed_delta = boundary_area_delta_ratio if touches_boundary else max_area_delta_ratio
            if area_delta_ratio > allowed_delta:
                invalid_keys.update(moved_keys)
        if not invalid_keys:
            break
        for key in invalid_keys:
            if key in snap_map:
                del snap_map[key]
                changed = True
        if not changed:
            break

    snapped: list[Polygon] = []
    for poly in cells:
        cand, _, _ = _snap_polygon_with_map(poly, mask, snap_map)
        snapped.append(cand)
    return snapped


def _cleanup_ring_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    clean: list[tuple[float, float]] = []
    for p in points:
        pt = (float(p[0]), float(p[1]))
        if not clean or math.dist(clean[-1], pt) > 1e-6:
            clean.append(pt)
    if len(clean) >= 2 and math.dist(clean[0], clean[-1]) < 1e-6:
        clean.pop()
    return clean


def _point_on_mask_boundary(pt: tuple[float, float], mask: Polygon, tol: float = 1.0) -> bool:
    return mask.boundary.distance(Point(float(pt[0]), float(pt[1]))) <= tol


def _polygon_from_points_with_mask(
    points: list[tuple[float, float]], mask: Polygon
) -> Polygon | None:
    clean = _cleanup_ring_points(points)
    if len(clean) < 3:
        return None
    try:
        cand = Polygon(clean).buffer(0).intersection(mask).buffer(0)
    except Exception:
        return None
    if cand.is_empty:
        return None
    if isinstance(cand, MultiPolygon):
        cand = max(cand.geoms, key=lambda g: g.area, default=None)
    if cand is None or cand.is_empty or not isinstance(cand, Polygon) or cand.area <= 1e-6:
        return None
    return cand


def _assign_regions_to_snapped_cells(
    region_polys: list[Polygon], snapped_polys: list[Polygon]
) -> dict[str, list[int]]:
    snap_region_map: dict[str, list[int]] = {str(i): [] for i in range(len(snapped_polys))}
    snap_centroids = [poly.representative_point() for poly in snapped_polys]
    for rid, region_poly in enumerate(region_polys):
        if region_poly.is_empty:
            continue
        rp = region_poly.representative_point()
        assigned = None
        for zid, snap_poly in enumerate(snapped_polys):
            if snap_poly.covers(rp):
                assigned = zid
                break
        if assigned is None:
            best_overlap = -1.0
            for zid, snap_poly in enumerate(snapped_polys):
                try:
                    overlap = region_poly.intersection(snap_poly).area
                except Exception:
                    overlap = 0.0
                if overlap > best_overlap:
                    best_overlap = overlap
                    assigned = zid
        if assigned is None and snap_centroids:
            best_dist = float("inf")
            for zid, pt in enumerate(snap_centroids):
                dx = float(pt.x) - float(rp.x)
                dy = float(pt.y) - float(rp.y)
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    assigned = zid
        if assigned is not None:
            snap_region_map[str(int(assigned))].append(int(rid))
    return snap_region_map


def _remove_vertex_key_from_points(
    points: list[tuple[float, float]],
    key: tuple[float, float],
) -> list[tuple[float, float]]:
    return [p for p in points if _vertex_key(p) != key]


def _boundary_vertex_score(
    points: list[tuple[float, float]],
    key: tuple[float, float],
) -> float:
    n = len(points)
    if n < 3:
        return float("inf")
    best = float("inf")
    for i, p in enumerate(points):
        if _vertex_key(p) != key:
            continue
        prev = points[(i - 1) % n]
        cur = points[i]
        nxt = points[(i + 1) % n]
        score = abs(
            (prev[0] * (cur[1] - nxt[1]) + cur[0] * (nxt[1] - prev[1]) + nxt[0] * (prev[1] - cur[1]))
            * 0.5
        )
        if score < best:
            best = score
    return best


def _simplify_boundary_cells_consistent(
    snapped: list[Polygon],
    original_cells: list[Polygon],
    mask: Polygon,
    max_area_delta_ratio: float = 0.2,
) -> list[Polygon]:
    current_pts: list[list[tuple[float, float]]] = [_cleanup_ring_points(_poly_to_vertices(poly)) for poly in snapped]
    original_areas = [max(float(poly.area), 1e-6) for poly in original_cells]
    usage: dict[tuple[float, float], set[int]] = defaultdict(set)
    for cid, pts in enumerate(current_pts):
        for p in pts:
            usage[_vertex_key(p)].add(cid)

    boundary_candidates: list[tuple[float, tuple[float, float]]] = []
    for key, cell_ids in usage.items():
        if not _point_on_mask_boundary(key, mask, tol=1.0):
            continue
        score = 0.0
        valid = False
        for cid in cell_ids:
            pts = current_pts[cid]
            if len(pts) <= 3:
                continue
            score += _boundary_vertex_score(pts, key)
            valid = True
        if valid:
            boundary_candidates.append((score, key))

    boundary_candidates.sort(key=lambda item: item[0])
    removed_count = 0
    max_removals = 256
    for _score, key in boundary_candidates:
        if removed_count >= max_removals:
            break
        cell_ids = sorted(usage.get(key, set()))
        if not cell_ids:
            continue
        next_polys: dict[int, Polygon] = {}
        ok = True
        for cid in cell_ids:
            pts = current_pts[cid]
            if len(pts) <= 3:
                ok = False
                break
            next_pts = _remove_vertex_key_from_points(pts, key)
            poly = _polygon_from_points_with_mask(next_pts, mask)
            if poly is None:
                ok = False
                break
            area_delta_ratio = abs(float(poly.area) - original_areas[cid]) / original_areas[cid]
            if area_delta_ratio > max_area_delta_ratio:
                ok = False
                break
            next_polys[cid] = poly
        if not ok:
            continue
        for cid, poly in next_polys.items():
            current_pts[cid] = _cleanup_ring_points(_poly_to_vertices(poly))
        removed_count += 1

    out: list[Polygon] = []
    for cid, pts in enumerate(current_pts):
        poly = _polygon_from_points_with_mask(pts, mask)
        out.append(poly if poly is not None else snapped[cid])
    return out


def _iter_polygons(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    if hasattr(geom, "geoms"):
        out: list[Polygon] = []
        for g in geom.geoms:
            out.extend(_iter_polygons(g))
        return out
    return []


def _fill_mask_gaps(snapped: list[Polygon], mask: Polygon) -> list[Polygon]:
    if not snapped:
        return []
    current = [poly.buffer(0) for poly in snapped]
    union = unary_union(current).buffer(0)
    gaps_geom = mask.buffer(0).difference(union).buffer(0)
    gaps = [g for g in _iter_polygons(gaps_geom) if g.area > 1e-6]
    if not gaps:
        return current

    for gap in gaps:
        best_idx = None
        best_score = -1.0
        for idx, poly in enumerate(current):
            try:
                score = poly.boundary.intersection(gap.boundary).length
            except Exception:
                score = 0.0
            if score > best_score + 1e-9:
                best_score = score
                best_idx = idx
        if best_idx is None:
            continue
        merged = current[best_idx].union(gap).buffer(0).intersection(mask).buffer(0)
        if isinstance(merged, MultiPolygon):
            merged = max(merged.geoms, key=lambda g: g.area, default=current[best_idx])
        if isinstance(merged, Polygon) and not merged.is_empty and merged.area > 1e-6:
            current[best_idx] = merged
    return current


def build_source_voronoi(source_path: Path, count: int | None = None, relax: int = 2, seed: int = 7) -> dict:
    source_count = max(1, int(count or config.TARGET_ZONES))
    source_relax = max(0, int(relax))
    source_seed = int(seed)

    vb, mask, targets = _parse_source(source_path.read_bytes())
    rng = random.Random(source_seed)
    pts = _random_points_in_polygon(mask, source_count, rng)

    for _ in range(source_relax):
        vor = Voronoi(pts)
        regions, vertices = _voronoi_finite_polygons_2d(vor)
        new_pts = []
        for region in regions:
            poly = Polygon(vertices[region]).buffer(0).intersection(mask).buffer(0)
            if poly.is_empty:
                continue
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)
            if poly.area > 1e-6:
                new_pts.append(_representative(poly))
        pts = np.array(new_pts)
        if len(pts) < source_count:
            extra = _random_points_in_polygon(mask, source_count - len(pts), rng)
            pts = np.vstack([pts, extra])

    vor = Voronoi(pts)
    regions, vertices = _voronoi_finite_polygons_2d(vor)
    cells = []
    for region in regions:
        poly = Polygon(vertices[region]).buffer(0).intersection(mask).buffer(0)
        if poly.is_empty:
            continue
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        if poly.area > 1e-6 and mask.contains(poly.representative_point()):
            cells.append(poly)
    cells = cells[:source_count]
    snapped = _snap_cells_consistent(
        cells,
        targets,
        mask,
        max_dist=14.0,
        max_area_delta_ratio=0.28,
        boundary_area_delta_ratio=0.2,
    )
    snapped = _simplify_boundary_cells_consistent(
        snapped,
        cells,
        mask,
        max_area_delta_ratio=0.2,
    )
    snapped = _fill_mask_gaps(snapped, mask)

    def poly_to_vertices(poly: Polygon):
        return [[float(x), float(y)] for x, y in _poly_to_vertices(poly)]

    return {
        "viewBox": [float(v) for v in vb],
        "mask": poly_to_vertices(mask),
        "cells": [poly_to_vertices(poly) for poly in cells],
        "snapped_cells": [poly_to_vertices(poly) for poly in snapped],
        "count": len(cells),
        "target_count": source_count,
        "relax": source_relax,
        "seed": source_seed,
        "snap_max_dist": 14.0,
        "snap_max_area_delta_ratio": 0.28,
        "snap_boundary_area_delta_ratio": 0.2,
    }


def _is_near_canvas_rect(
    pts: list[tuple[float, float]],
    canvas: tuple[int, int],
    margin: float = 2.0,
    min_area_ratio: float = 0.9,
) -> bool:
    if len(pts) != 4:
        return False
    poly = Polygon(pts)
    if poly.is_empty or poly.area <= 0:
        return False
    w, h = float(canvas[0]), float(canvas[1])
    canvas_area = max(w * h, 1.0)
    if poly.area < canvas_area * min_area_ratio:
        return False
    minx, miny, maxx, maxy = poly.bounds
    if abs(minx - 0.0) > margin or abs(miny - 0.0) > margin:
        return False
    if abs(maxx - w) > margin or abs(maxy - h) > margin:
        return False
    xs = sorted({round(float(x), 3) for x, _ in pts})
    ys = sorted({round(float(y), 3) for _, y in pts})
    return len(xs) <= 2 and len(ys) <= 2


def _load_source_segments(svg_path: Path) -> tuple[list[LineString], tuple[int, int]]:
    root = ET.parse(svg_path).getroot()
    canvas = svg_utils._get_canvas_size(root, 1.0)
    lines: list[LineString] = []
    for gtype, pts in svg_utils._iter_geometry(root):
        if gtype == "polyline":
            for a, b in zip(pts, pts[1:]):
                if a != b:
                    lines.append(LineString([a, b]))
        elif gtype == "polygon":
            for a, b in zip(pts, pts[1:] + [pts[0]]):
                if a != b:
                    lines.append(LineString([a, b]))
    return lines, canvas


def _load_cached_source_segments(
    svg_path: Path,
    cached_nodes: list[dict] | None,
    cached_segments: list[list[int]] | None,
) -> tuple[list[LineString], tuple[int, int]]:
    root = ET.parse(svg_path).getroot()
    canvas = svg_utils._get_canvas_size(root, 1.0)
    if not cached_nodes or not cached_segments:
        return [], canvas
    lines: list[LineString] = []
    for seg in cached_segments:
        if not isinstance(seg, (list, tuple)) or len(seg) < 2:
            continue
        try:
            ai = int(seg[0])
            bi = int(seg[1])
        except Exception:
            continue
        if ai < 0 or bi < 0 or ai >= len(cached_nodes) or bi >= len(cached_nodes):
            continue
        a_raw = cached_nodes[ai]
        b_raw = cached_nodes[bi]
        try:
            a = (float(a_raw["x"]), float(a_raw["y"]))
            b = (float(b_raw["x"]), float(b_raw["y"]))
        except Exception:
            continue
        if abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9:
            continue
        lines.append(LineString([a, b]))
    return lines, canvas


def _project_point_to_segment(
    pt: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> tuple[float, float] | None:
    ax, ay = a
    bx, by = b
    px, py = pt
    dx = bx - ax
    dy = by - ay
    ll = dx * dx + dy * dy
    if ll <= 1e-9:
        return None
    t = ((px - ax) * dx + (py - ay) * dy) / ll
    if t <= 1e-6 or t >= 1.0 - 1e-6:
        return None
    return (ax + t * dx, ay + t * dy)


def _project_point_to_segment_clamped(
    pt: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> tuple[float, float] | None:
    ax, ay = a
    bx, by = b
    px, py = pt
    dx = bx - ax
    dy = by - ay
    ll = dx * dx + dy * dy
    if ll <= 1e-9:
        return None
    t = ((px - ax) * dx + (py - ay) * dy) / ll
    t = max(0.0, min(1.0, t))
    return (ax + t * dx, ay + t * dy)


def _dedupe_poly_pts(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not pts:
        return []
    out: list[tuple[float, float]] = []
    for p in pts:
        if not out or abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) < 1e-9 and abs(out[0][1] - out[-1][1]) < 1e-9:
        out = out[:-1]
    return out


def _project_region_vertices_to_zone_boundary(
    regions: list[list[list[float]]],
    snapped_cells: list[list[list[float]]],
    snap_region_map: dict[str, list[int]],
    radius: float,
    min_area: float,
) -> list[list[list[float]]]:
    if radius <= 0 or not regions or not snapped_cells or not snap_region_map:
        return regions

    out: list[list[list[float]]] = []
    for rid, raw_pts in enumerate(regions):
        pts = [(float(x), float(y)) for x, y in raw_pts]
        zid: int | None = None
        for k, ids in snap_region_map.items():
            if rid in ids:
                try:
                    zid = int(k)
                except Exception:
                    zid = None
                break
        if zid is None or zid < 0 or zid >= len(snapped_cells):
            out.append([[float(x), float(y)] for x, y in pts])
            continue

        zone = snapped_cells[zid]
        if not zone or len(zone) < 3:
            out.append([[float(x), float(y)] for x, y in pts])
            continue
        zpts = [(float(x), float(y)) for x, y in zone]
        segs = list(zip(zpts, zpts[1:] + zpts[:1]))

        moved: list[tuple[float, float]] = []
        r2 = radius * radius
        for p in pts:
            best = p
            best_d2 = r2
            found = False
            for a, b in segs:
                proj = _project_point_to_segment_clamped(p, a, b)
                if proj is None:
                    continue
                dx = proj[0] - p[0]
                dy = proj[1] - p[1]
                d2 = dx * dx + dy * dy
                if d2 <= best_d2:
                    best_d2 = d2
                    best = proj
                    found = True
            moved.append(best if found else p)

        moved = _dedupe_poly_pts(moved)
        if len(moved) < 3:
            continue
        poly = _coerce_polygon(Polygon(moved), allow_largest=True)
        if poly is None or poly.area < min_area:
            continue
        coords = list(poly.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        out.append([[float(x), float(y)] for x, y in coords])
    return out


def _polygon_min_width(poly: Polygon) -> float:
    try:
        rect = poly.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)
    except Exception:
        return float("inf")
    if len(coords) < 4:
        return float("inf")
    lengths: list[float] = []
    for a, b in zip(coords, coords[1:]):
        lengths.append(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))
    return min(lengths) if lengths else float("inf")


def _merge_thin_regions_exact(
    regions: list[list[list[float]]],
    snap_region_map: dict[str, list[int]],
    snapped_cells: list[list[list[float]]],
    max_width: float,
    max_area: float,
    *,
    require_boundary_touch: bool,
) -> list[list[list[float]]]:
    if not regions or not snap_region_map or not snapped_cells:
        return regions

    polys: list[Polygon | None] = []
    for pts in regions:
        try:
            p = _coerce_polygon(Polygon([(float(x), float(y)) for x, y in pts]), allow_largest=True)
        except Exception:
            p = None
        polys.append(p)

    for zid_str, rid_list in snap_region_map.items():
        try:
            zid = int(zid_str)
        except Exception:
            continue
        if zid < 0 or zid >= len(snapped_cells):
            continue
        zpts = [(float(x), float(y)) for x, y in snapped_cells[zid]]
        if len(zpts) < 3:
            continue
        zline = LineString(zpts + [zpts[0]])
        zone_ids = [int(rid) for rid in rid_list if 0 <= int(rid) < len(polys)]
        if len(zone_ids) < 2:
            continue

        changed = True
        while changed:
            changed = False
            for rid in list(zone_ids):
                rp = polys[rid]
                if rp is None:
                    continue
                try:
                    touch_len = float(rp.boundary.intersection(zline).length)
                except Exception:
                    touch_len = 0.0
                if require_boundary_touch and touch_len <= 1e-7:
                    continue
                width = _polygon_min_width(rp)
                if not (rp.area <= max_area or width <= max_width):
                    continue

                best_nid = None
                best_shared = 0.0
                for nid in zone_ids:
                    if nid == rid:
                        continue
                    npoly = polys[nid]
                    if npoly is None:
                        continue
                    try:
                        shared = float(rp.boundary.intersection(npoly.boundary).length)
                    except Exception:
                        shared = 0.0
                    if shared > best_shared:
                        best_shared = shared
                        best_nid = nid
                if best_nid is None or best_shared <= 1e-7:
                    continue

                npoly = polys[best_nid]
                if npoly is None:
                    continue
                try:
                    # Exact topological union, no buffer approximation.
                    merged_geom = npoly.union(rp)
                except Exception:
                    continue
                merged_poly = _coerce_polygon(merged_geom, allow_largest=True)
                if merged_poly is None:
                    continue
                polys[best_nid] = merged_poly
                polys[rid] = None
                changed = True
                break

    out: list[list[list[float]]] = []
    for p in polys:
        if p is None or p.is_empty or p.area <= 1e-6:
            continue
        coords = list(p.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) < 3:
            continue
        out.append([[float(x), float(y)] for x, y in coords])
    return out


def _snap_source_vertices_to_zone_segments(
    source_lines: list[LineString],
    snapped_cells: list[list[list[float]]],
    max_dist: float = 4.0,
) -> tuple[list[LineString], list[LineString]]:
    if not source_lines or not snapped_cells:
        return source_lines, []

    source_segments: list[tuple[int, tuple[float, float], tuple[float, float]]] = []
    for ln in source_lines:
        coords = list(ln.coords)
        if len(coords) < 2:
            continue
        a = (float(coords[0][0]), float(coords[0][1]))
        b = (float(coords[-1][0]), float(coords[-1][1]))
        if a != b:
            source_segments.append((len(source_segments), a, b))

    snap_segments: list[tuple[int, tuple[float, float], tuple[float, float]]] = []
    for poly in snapped_cells:
        if len(poly) < 2:
            continue
        pts = [(float(x), float(y)) for x, y in poly]
        for a, b in zip(pts, pts[1:] + [pts[0]]):
            if a != b:
                snap_segments.append((len(snap_segments), a, b))

    def _pt_key(pt: tuple[float, float]) -> tuple[int, int]:
        return (int(round(pt[0] * 1000.0)), int(round(pt[1] * 1000.0)))

    moved_source: dict[tuple[int, int], tuple[float, float]] = {}
    moved_snap: dict[tuple[int, int], tuple[float, float]] = {}
    source_segment_splits: dict[int, list[tuple[float, float]]] = {}
    snap_segment_splits: dict[int, list[tuple[float, float]]] = {}

    def _snap_vertices(
        vertices: list[tuple[float, float]],
        target_segments: list[tuple[int, tuple[float, float], tuple[float, float]]],
        moved_map: dict[tuple[int, int], tuple[float, float]],
        split_map: dict[int, list[tuple[float, float]]],
    ) -> None:
        for p in vertices:
            key = _pt_key(p)
            if key in moved_map:
                continue
            best_proj = None
            best_d2 = max_dist * max_dist
            best_seg_idx = None
            for seg_idx, a, b in target_segments:
                proj = _project_point_to_segment(p, a, b)
                if proj is None:
                    continue
                ddx = proj[0] - p[0]
                ddy = proj[1] - p[1]
                d2 = ddx * ddx + ddy * ddy
                if d2 <= best_d2:
                    best_d2 = d2
                    best_proj = proj
                    best_seg_idx = seg_idx
            moved_map[key] = best_proj if best_proj is not None else p
            if best_proj is not None and best_seg_idx is not None:
                split_map.setdefault(best_seg_idx, []).append(best_proj)

    source_vertices: list[tuple[float, float]] = []
    for _, a, b in source_segments:
        source_vertices.extend([a, b])
    snap_vertices: list[tuple[float, float]] = []
    for _, a, b in snap_segments:
        snap_vertices.extend([a, b])

    _snap_vertices(source_vertices, snap_segments, moved_source, snap_segment_splits)
    _snap_vertices(snap_vertices, source_segments, moved_snap, source_segment_splits)

    def _split_segments(
        segments: list[tuple[int, tuple[float, float], tuple[float, float]]],
        moved_map: dict[tuple[int, int], tuple[float, float]],
        split_map: dict[int, list[tuple[float, float]]],
    ) -> list[LineString]:
        out: list[LineString] = []
        for seg_idx, a0, b0 in segments:
            a = moved_map.get(_pt_key(a0), a0)
            b = moved_map.get(_pt_key(b0), b0)
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            ll = dx * dx + dy * dy
            if ll <= 1e-9:
                continue
            pts = [a]
            pts.extend(split_map.get(seg_idx, []))
            pts.append(b)
            pts = sorted(
                pts,
                key=lambda p: ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / ll,
            )
            deduped: list[tuple[float, float]] = []
            for p in pts:
                if not deduped or abs(p[0] - deduped[-1][0]) > 1e-9 or abs(p[1] - deduped[-1][1]) > 1e-9:
                    deduped.append(p)
            for p0, p1 in zip(deduped, deduped[1:]):
                if abs(p0[0] - p1[0]) < 1e-9 and abs(p0[1] - p1[1]) < 1e-9:
                    continue
                out.append(LineString([p0, p1]))
        return out

    snapped_source_lines = _split_segments(source_segments, moved_source, source_segment_splits)
    snapped_snap_lines = _split_segments(snap_segments, moved_snap, snap_segment_splits)
    return snapped_source_lines, snapped_snap_lines


def build_source_region_scene(
    source_path: Path,
    count: int | None = None,
    relax: int = 2,
    seed: int = 7,
    cached_nodes: list[dict] | None = None,
    cached_segments: list[list[int]] | None = None,
    cached_voronoi: dict | None = None,
) -> dict:
    config.SVG_PATH = source_path
    voronoi = cached_voronoi or build_source_voronoi(source_path, count=count, relax=relax, seed=seed)
    if cached_nodes and cached_segments:
        source_lines, canvas = _load_cached_source_segments(source_path, cached_nodes, cached_segments)
    else:
        source_lines, canvas = _load_source_segments(source_path)
    snapped_cells = voronoi.get("snapped_cells", [])
    snap_lines: list[LineString] = []
    for poly in snapped_cells:
        if not poly or len(poly) < 3:
            continue
        pts = [(float(x), float(y)) for x, y in poly]
        for a, b in zip(pts, pts[1:] + pts[:1]):
            if abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9:
                continue
            snap_lines.append(LineString([a, b]))
    # Snap near-boundary source vertices onto zone boundary before polygonize.
    # This suppresses ultra-thin sliver regions that would otherwise block bleed.
    snap_threshold = max(0.0, float(config.PACK_BLEED))
    merged_lines: list[LineString]
    if snap_threshold > 0.0 and snapped_cells:
        snapped_source_lines, snapped_snap_lines = _snap_source_vertices_to_zone_segments(
            source_lines,
            snapped_cells,
            max_dist=snap_threshold,
        )
        if snapped_source_lines:
            merged_lines = list(snapped_source_lines)
            if snapped_snap_lines:
                merged_lines.extend(snapped_snap_lines)
            else:
                merged_lines.extend(snap_lines)
        else:
            merged_lines = list(source_lines)
            merged_lines.extend(snap_lines)
    else:
        merged_lines = list(source_lines)
        merged_lines.extend(snap_lines)

    merged = unary_union(merged_lines)
    poly_pts, _border_pts = polygonize_full(merged)[:2]
    regions: list[list[list[float]]] = []
    for geom in (poly_pts.geoms if hasattr(poly_pts, "geoms") else [poly_pts]):
        if geom.is_empty or geom.geom_type != "Polygon":
            continue
        coords = list(geom.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        pts = [(float(x), float(y)) for x, y in coords]
        if len(pts) < 3 or _is_near_canvas_rect(pts, canvas):
            continue
        poly = Polygon(pts)
        if poly.is_empty or poly.area <= 1e-6:
            continue
        regions.append([[float(x), float(y)] for x, y in pts])

    snapped_polys = []
    for poly in voronoi.get("snapped_cells", []):
        if len(poly) < 3:
            continue
        clean = _coerce_polygon(Polygon(poly), allow_largest=True)
        if clean is not None:
            snapped_polys.append(clean)
    region_polys = []
    for pts in regions:
        try:
            poly = _coerce_polygon(Polygon([(float(x), float(y)) for x, y in pts]), allow_largest=True)
        except Exception:
            poly = None
        region_polys.append(poly)
    snap_region_map = _assign_regions_to_snapped_cells(region_polys, snapped_polys)

    # Project near-boundary region vertices onto zone boundary, then drop tiny slivers.
    projection_radius = max(0.0, float(config.PACK_BLEED))
    if projection_radius > 0.0:
        min_area = max(1e-3, projection_radius * projection_radius * 0.05)
        regions = _project_region_vertices_to_zone_boundary(
            regions,
            voronoi.get("snapped_cells", []) or [],
            snap_region_map,
            projection_radius,
            min_area,
        )
        region_polys = []
        for pts in regions:
            try:
                poly = _coerce_polygon(Polygon([(float(x), float(y)) for x, y in pts]), allow_largest=True)
            except Exception:
                poly = None
            region_polys.append(poly)
        snap_region_map = _assign_regions_to_snapped_cells(region_polys, snapped_polys)
        # Pass 1: merge very thin slivers touching zone boundary.
        regions = _merge_thin_regions_exact(
            regions,
            snap_region_map,
            voronoi.get("snapped_cells", []) or [],
            max_width=max(0.5, projection_radius * 0.9),
            max_area=max(1e-3, projection_radius * projection_radius * 0.08),
            require_boundary_touch=True,
        )
        region_polys = []
        for pts in regions:
            try:
                poly = _coerce_polygon(Polygon([(float(x), float(y)) for x, y in pts]), allow_largest=True)
            except Exception:
                poly = None
            region_polys.append(poly)
        snap_region_map = _assign_regions_to_snapped_cells(region_polys, snapped_polys)

        # Pass 2: merge remaining ultra-thin internal slivers via exact union.
        regions = _merge_thin_regions_exact(
            regions,
            snap_region_map,
            voronoi.get("snapped_cells", []) or [],
            max_width=max(0.45, projection_radius * 0.65),
            max_area=max(1e-3, projection_radius * projection_radius * 0.05),
            require_boundary_touch=False,
        )
        region_polys = []
        for pts in regions:
            try:
                poly = _coerce_polygon(Polygon([(float(x), float(y)) for x, y in pts]), allow_largest=True)
            except Exception:
                poly = None
            region_polys.append(poly)
        snap_region_map = _assign_regions_to_snapped_cells(region_polys, snapped_polys)

    colors_bgr, _ = geometry.compute_region_colors(
        [[(float(x), float(y)) for x, y in pts] for pts in regions],
        canvas,
    )
    region_colors = [f"#{r:02x}{g:02x}{b:02x}" for (b, g, r) in colors_bgr]
    return {
        "canvas": {"w": int(canvas[0]), "h": int(canvas[1])},
        "regions": regions,
        "region_colors": region_colors,
        "snap_region_map": snap_region_map,
        "source_name": source_path.name,
        "voronoi": voronoi,
    }
