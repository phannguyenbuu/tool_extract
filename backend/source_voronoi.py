from __future__ import annotations

import math
import random
import re
from collections import defaultdict
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Iterable

import numpy as np
from scipy.spatial import Voronoi
from shapely import intersection as shapely_intersection
from shapely.errors import GEOSException
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, polygonize_full, unary_union
from shapely.validation import make_valid

import config
import geometry
import svg_utils

SOURCE_VORONOI_VERSION = 18
FALLBACK_REGION_COLORS = [
    "#ff6b6b",
    "#4ecdc4",
    "#ffe66d",
    "#5dade2",
    "#af7ac5",
    "#58d68d",
    "#f5b041",
    "#ec7063",
    "#85c1e9",
    "#f7dc6f",
    "#73c6b6",
    "#bb8fce",
]


def _is_hex_color(value: str | None) -> bool:
    return isinstance(value, str) and re.match(r"^#[0-9a-fA-F]{6}$", value or "") is not None


def _repair_region_colors_by_adjacency(region_polys: list[Polygon], colors: list[str]) -> list[str]:
    n = len(region_polys)
    if n <= 0:
        return []
    fixed = [c.lower() if _is_hex_color(c) else "" for c in (colors or [])[:n]]
    if len(fixed) < n:
        fixed.extend([""] * (n - len(fixed)))

    neigh: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        pi = region_polys[i]
        if pi is None or pi.is_empty:
            continue
        for j in range(i + 1, n):
            pj = region_polys[j]
            if pj is None or pj.is_empty:
                continue
            try:
                inter = pi.boundary.intersection(pj.boundary)
                score = float(getattr(inter, "length", 0.0))
            except GEOSException:
                score = 0.0
            if score <= 1e-6:
                continue
            neigh[i].append(j)
            neigh[j].append(i)

    # Propagate colors from adjacent regions first.
    for _ in range(4):
        changed = False
        for i in range(n):
            if fixed[i]:
                continue
            best_color = ""
            best_len = -1.0
            pi = region_polys[i]
            if pi is None or pi.is_empty:
                continue
            for j in neigh[i]:
                c = fixed[j]
                if not c:
                    continue
                pj = region_polys[j]
                if pj is None or pj.is_empty:
                    continue
                try:
                    edge_len = float(pi.boundary.intersection(pj.boundary).length)
                except GEOSException:
                    edge_len = 0.0
                if edge_len > best_len:
                    best_len = edge_len
                    best_color = c
            if best_color:
                fixed[i] = best_color
                changed = True
        if not changed:
            break

    # Final fallback if a region still has no valid adjacent color.
    for i in range(n):
        if fixed[i]:
            continue
        fixed[i] = FALLBACK_REGION_COLORS[i % len(FALLBACK_REGION_COLORS)]

    return fixed


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
        if candidate is None or candidate.is_empty:
            continue
        if isinstance(candidate, Polygon):
            if candidate.area <= 1e-6:
                continue
            if not candidate.is_valid:
                try:
                    repaired = candidate.buffer(0)
                except Exception:
                    repaired = None
                if isinstance(repaired, Polygon) and not repaired.is_empty and repaired.area > 1e-6:
                    return repaired
                if hasattr(repaired, "geoms"):
                    pieces = [g for g in repaired.geoms if isinstance(g, Polygon) and not g.is_empty and g.area > 1e-6]
                    if pieces:
                        return max(pieces, key=lambda g: g.area) if allow_largest else pieces[0]
                continue
            return candidate
        elif hasattr(candidate, "geoms"):
            pieces = []
            for g in candidate.geoms:
                if not isinstance(g, Polygon) or g.is_empty or g.area <= 1e-6:
                    continue
                if not g.is_valid:
                    try:
                        repaired = g.buffer(0)
                    except Exception:
                        repaired = None
                    if isinstance(repaired, Polygon) and not repaired.is_empty and repaired.area > 1e-6:
                        pieces.append(repaired)
                    elif hasattr(repaired, "geoms"):
                        pieces.extend(
                            part for part in repaired.geoms if isinstance(part, Polygon) and not part.is_empty and part.area > 1e-6
                        )
                    continue
                pieces.append(g)
            if not pieces:
                continue
            return max(pieces, key=lambda g: g.area) if allow_largest else pieces[0]
    return None


def _parse_path_d(d: str):
    tokens = re.findall(r"[AaCcHhLlMmQqSsTtVvZz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    i = 0
    cmd = None
    x = y = 0.0
    start = None
    pts: List[Tuple[float, float]] = []
    subpaths = []

    def flush():
        nonlocal pts
        if len(pts) >= 2:
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
            continue
        if cmd is None:
            i += 1
            continue
        if cmd in "Mm":
            nx, ny = float(tok), float(tokens[i + 1]); i += 2
            if cmd == "m": nx += x; ny += y
            if pts: flush()
            pts = [(nx, ny)]
            x, y = nx, ny
            start = (x, y)
            cmd = "l" if cmd == "m" else "L"
        elif cmd in "Ll":
            nx, ny = float(tok), float(tokens[i + 1]); i += 2
            if cmd == "l": nx += x; ny += y
            append_point(nx, ny)
        elif cmd in "Hh":
            nx = float(tok); i += 1
            nx = nx + x if cmd == "h" else nx
            append_point(nx, y)
        elif cmd in "Vv":
            ny = float(tok); i += 1
            ny = ny + y if cmd == "v" else ny
            append_point(x, ny)
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
    raw_vertices: List[Tuple[float, float]] = []
    parent_map = {c: p for p in root.iter() for c in p}
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag in {"polygon", "polyline", "path", "line"}: # Exclude 'rect' from seeds to avoid corner bias
            m = svg_utils._get_element_transform(elem, parent_map)
            if tag == "polygon":
                pts = svg_utils._apply_matrix(_parse_points_attr(elem.get("points", "")), m)
                raw_vertices.extend(pts)
            elif tag == "path":
                for pts_raw in _parse_path_d(elem.get("d", "")):
                    raw_vertices.extend(svg_utils._apply_matrix(pts_raw, m))
    uniq: List[Tuple[float, float]] = []
    seen = set()
    for x, y in raw_vertices:
        key = (round(x, 3), round(y, 3))
        if key not in seen:
            seen.add(key); uniq.append((float(x), float(y)))
    return vb, uniq


def _random_points_in_polygon(poly: Polygon, count: int, rng: random.Random):
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    attempts = 0
    while len(pts) < count and attempts < count * 5000:
        attempts += 1
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if poly.contains(Point(x, y)):
            pts.append((x, y))
    if len(pts) < count:
        # Fallback to random uniform in bbox if sampling is too hard
        while len(pts) < count:
            pts.append((rng.uniform(minx, maxx), rng.uniform(miny, maxy)))
    return np.array(pts)


def _dedup_xy_points(
    pts: Iterable[tuple[float, float]],
    *,
    digits: int = 3,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for x, y in pts:
        key = (round(float(x), digits), round(float(y), digits))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(x), float(y)))
    return out


def _collect_source_seed_points(
    source_lines: list[LineString],
    mask: Polygon,
) -> list[tuple[float, float]]:
    if not source_lines or mask is None or mask.is_empty:
        return []
    candidates: list[tuple[float, float]] = []
    inner_mask = mask.buffer(-0.25)
    boundary_keys = {(round(x, 2), round(y, 2)) for x, y in _poly_to_vertices(mask)}
    for ln in source_lines:
        coords = list(ln.coords)
        if len(coords) < 2:
            continue
        for x, y in (coords[0], coords[-1]):
            pt = (float(x), float(y))
            if (round(pt[0], 2), round(pt[1], 2)) in boundary_keys:
                continue
            try:
                if inner_mask.contains(Point(pt)):
                    candidates.append(pt)
            except Exception:
                continue
    return _dedup_xy_points(candidates, digits=3)


def _farthest_point_sample(
    pts: list[tuple[float, float]],
    target: int,
    mask: Polygon,
) -> list[tuple[float, float]]:
    if target <= 0 or not pts:
        return []
    if len(pts) <= target:
        return pts[:]
    arr = np.asarray(pts, dtype=float)
    minx, miny, maxx, maxy = mask.bounds
    center = np.array([(minx + maxx) * 0.5, (miny + maxy) * 0.5], dtype=float)
    d_center = np.sum((arr - center) ** 2, axis=1)
    first = int(np.argmin(d_center))
    chosen = [first]
    min_d2 = np.sum((arr - arr[first]) ** 2, axis=1)
    while len(chosen) < target:
        idx = int(np.argmax(min_d2))
        chosen.append(idx)
        d2 = np.sum((arr - arr[idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    return [tuple(map(float, arr[idx])) for idx in chosen]


def _voronoi_finite_polygons_2d(vor: Voronoi, radius=None):
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
            new_regions.append(vertices); continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0 or v1 < 0: v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0: continue
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


def _clean_poly_vertices(
    pts: List[Tuple[float, float]],
    *,
    min_seg: float = 1.25,
    round_snap: float = 0.0,
) -> List[Tuple[float, float]]:
    if len(pts) < 3:
        return []
    min_seg2 = min_seg * min_seg
    out: List[Tuple[float, float]] = []
    for x, y in pts:
        px = float(x)
        py = float(y)
        if round_snap > 0:
            px = round(px / round_snap) * round_snap
            py = round(py / round_snap) * round_snap
        p = (px, py)
        if not out:
            out.append(p)
            continue
        dx = p[0] - out[-1][0]
        dy = p[1] - out[-1][1]
        if dx * dx + dy * dy >= min_seg2:
            out.append(p)
    if len(out) > 1:
        dx = out[0][0] - out[-1][0]
        dy = out[0][1] - out[-1][1]
        if dx * dx + dy * dy < min_seg2:
            out.pop()
    return out if len(out) >= 3 else []


def _clean_polygon(
    poly: Polygon | None,
    *,
    simplify_tol: float = 0.9,
    min_seg: float = 1.25,
    round_snap: float = 0.0,
) -> Polygon | None:
    if poly is None or poly.is_empty:
        return None
    try:
        simplified = poly.simplify(simplify_tol, preserve_topology=True)
    except Exception:
        simplified = poly
    clean = _coerce_polygon(simplified, allow_largest=True) or _coerce_polygon(poly, allow_largest=True)
    if clean is None:
        return None
    pts = _clean_poly_vertices(_poly_to_vertices(clean), min_seg=min_seg, round_snap=round_snap)
    if len(pts) < 3:
        return None
    return _coerce_polygon(Polygon(pts), allow_largest=True)


def _polygon_boundary_lines(poly: Polygon | None) -> list[LineString]:
    if poly is None or poly.is_empty:
        return []
    coords = list(poly.exterior.coords)
    out: list[LineString] = []
    for i in range(len(coords) - 1):
        a = coords[i]
        b = coords[i + 1]
        if abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9:
            continue
        out.append(LineString([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))]))
    return out


def _graph_from_cells(
    cells: list[Polygon],
    *,
    digits: int = 4,
    min_seg: float = 0.18,
) -> tuple[list[list[float]], list[list[int]]]:
    key_to_id: dict[tuple[float, float], int] = {}
    vertices: list[list[float]] = []
    segments: list[list[int]] = []
    seen_edges: set[tuple[int, int]] = set()
    for poly in cells or []:
        pts = _clean_poly_vertices(_poly_to_vertices(poly), min_seg=min_seg)
        if len(pts) < 3:
            continue
        ids: list[int] = []
        for x, y in pts:
            key = (round(float(x), digits), round(float(y), digits))
            vid = key_to_id.get(key)
            if vid is None:
                vid = len(vertices)
                key_to_id[key] = vid
                vertices.append([float(x), float(y)])
            ids.append(vid)
        for i in range(len(ids)):
            a = ids[i]
            b = ids[(i + 1) % len(ids)]
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            segments.append([edge[0], edge[1]])
    return vertices, segments


def _collapse_short_graph_edges(
    vertices: list[list[float]],
    segments: list[list[int]],
    *,
    merge_dist: float = 0.9,
) -> tuple[list[list[float]], list[list[int]]]:
    if not vertices or not segments:
        return vertices, segments
    parent = list(range(len(vertices)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    merge_d2 = merge_dist * merge_dist
    for seg in segments:
        if not isinstance(seg, list) or len(seg) < 2:
            continue
        a = int(seg[0])
        b = int(seg[1])
        if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices) or a == b:
            continue
        ax, ay = vertices[a]
        bx, by = vertices[b]
        dx = float(ax) - float(bx)
        dy = float(ay) - float(by)
        if dx * dx + dy * dy <= merge_d2:
            union(a, b)

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(vertices)):
        groups[find(idx)].append(idx)

    remap: dict[int, int] = {}
    next_vertices: list[list[float]] = []
    for members in groups.values():
        sx = 0.0
        sy = 0.0
        for idx in members:
            sx += float(vertices[idx][0])
            sy += float(vertices[idx][1])
        rep_idx = len(next_vertices)
        next_vertices.append([sx / len(members), sy / len(members)])
        for idx in members:
            remap[idx] = rep_idx

    seen_edges: set[tuple[int, int]] = set()
    next_segments: list[list[int]] = []
    for seg in segments:
        if not isinstance(seg, list) or len(seg) < 2:
            continue
        a = remap.get(int(seg[0]))
        b = remap.get(int(seg[1]))
        if a is None or b is None or a == b:
            continue
        edge = (a, b) if a < b else (b, a)
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        next_segments.append([edge[0], edge[1]])
    return next_vertices, next_segments


def _round_weld_graph(
    vertices: list[list[float]],
    segments: list[list[int]],
    *,
    grid: float = 1.0,
) -> tuple[list[list[float]], list[list[int]]]:
    if not vertices or not segments or grid <= 0:
        return vertices, segments

    groups: dict[tuple[float, float], list[int]] = defaultdict(list)
    for idx, pt in enumerate(vertices):
        x = float(pt[0])
        y = float(pt[1])
        key = (round(x / grid) * grid, round(y / grid) * grid)
        groups[key].append(idx)

    remap: dict[int, int] = {}
    next_vertices: list[list[float]] = []
    for key, members in groups.items():
        rep_idx = len(next_vertices)
        next_vertices.append([float(key[0]), float(key[1])])
        for idx in members:
            remap[idx] = rep_idx

    seen_edges: set[tuple[int, int]] = set()
    next_segments: list[list[int]] = []
    for seg in segments:
        if not isinstance(seg, list) or len(seg) < 2:
            continue
        a = remap.get(int(seg[0]))
        b = remap.get(int(seg[1]))
        if a is None or b is None or a == b:
            continue
        edge = (a, b) if a < b else (b, a)
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        next_segments.append([edge[0], edge[1]])
    return next_vertices, next_segments


def _collapse_tiny_graph_cells(
    cells: list[Polygon],
    vertices: list[list[float]],
    segments: list[list[int]],
    *,
    digits: int = 4,
    min_seg: float = 0.18,
    max_area: float = 18.0,
    max_bbox_area: float = 32.0,
    max_short_side: float = 4.2,
) -> tuple[list[list[float]], list[list[int]]]:
    if not cells or not vertices or not segments:
        return vertices, segments

    parent = list(range(len(vertices)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    key_to_ids: dict[tuple[float, float], list[int]] = defaultdict(list)
    for idx, pt in enumerate(vertices):
        key = (round(float(pt[0]), digits), round(float(pt[1]), digits))
        key_to_ids[key].append(idx)

    for poly in cells or []:
        if poly is None or poly.is_empty:
            continue
        minx, miny, maxx, maxy = poly.bounds
        bw = max(0.0, float(maxx) - float(minx))
        bh = max(0.0, float(maxy) - float(miny))
        bbox_area = bw * bh
        short_side = min(bw, bh)
        is_tiny = poly.area <= max_area or bbox_area <= max_bbox_area or short_side <= max_short_side
        if not is_tiny:
            continue
        pts = _clean_poly_vertices(_poly_to_vertices(poly), min_seg=min_seg)
        if len(pts) < 3:
            continue
        ids: list[int] = []
        for x, y in pts:
            key = (round(float(x), digits), round(float(y), digits))
            matches = key_to_ids.get(key) or []
            if matches:
                ids.append(matches[0])
        if len(ids) < 2:
            continue
        base = ids[0]
        for vid in ids[1:]:
            union(base, vid)

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(vertices)):
        groups[find(idx)].append(idx)

    remap: dict[int, int] = {}
    next_vertices: list[list[float]] = []
    for members in groups.values():
        sx = 0.0
        sy = 0.0
        for idx in members:
            sx += float(vertices[idx][0])
            sy += float(vertices[idx][1])
        rep_idx = len(next_vertices)
        next_vertices.append([sx / len(members), sy / len(members)])
        for idx in members:
            remap[idx] = rep_idx

    seen_edges: set[tuple[int, int]] = set()
    next_segments: list[list[int]] = []
    for seg in segments:
        if not isinstance(seg, list) or len(seg) < 2:
            continue
        a = remap.get(int(seg[0]))
        b = remap.get(int(seg[1]))
        if a is None or b is None or a == b:
            continue
        edge = (a, b) if a < b else (b, a)
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        next_segments.append([edge[0], edge[1]])
    return next_vertices, next_segments


def _graph_lines(
    vertices: list[list[float]],
    segments: list[list[int]],
) -> list[LineString]:
    lines: list[LineString] = []
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for seg in segments or []:
        if not isinstance(seg, list) or len(seg) < 2:
            continue
        a = int(seg[0])
        b = int(seg[1])
        if a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices) or a == b:
            continue
        ax, ay = float(vertices[a][0]), float(vertices[a][1])
        bx, by = float(vertices[b][0]), float(vertices[b][1])
        key_a = (round(ax, 4), round(ay, 4))
        key_b = (round(bx, 4), round(by, 4))
        key = (key_a, key_b) if key_a <= key_b else (key_b, key_a)
        if key in seen:
            continue
        seen.add(key)
        lines.append(LineString([(ax, ay), (bx, by)]))
    return lines


def _cells_from_graph(
    mask: Polygon,
    vertices: list[list[float]],
    segments: list[list[int]],
    *,
    simplify_tol: float = 0.18,
    min_seg: float = 0.35,
) -> list[Polygon]:
    if mask is None or mask.is_empty:
        return []
    lines = _graph_lines(vertices, segments)
    if not lines:
        return []
    merged = unary_union(lines + _polygon_boundary_lines(mask))
    poly_pts, _ = polygonize_full(merged)[:2]
    cells: list[Polygon] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    geoms = poly_pts.geoms if hasattr(poly_pts, "geoms") else [poly_pts]
    for geom in geoms:
        if geom.is_empty or geom.geom_type != "Polygon":
            continue
        clipped = _coerce_polygon(geom.intersection(mask), allow_largest=True)
        if clipped is None or clipped.is_empty or clipped.area <= 1e-4:
            continue
        clean = _clean_polygon(clipped, simplify_tol=simplify_tol, min_seg=min_seg)
        if clean is None or clean.area <= 1e-4:
            continue
        key = tuple((round(x, 4), round(y, 4)) for x, y in _poly_to_vertices(clean))
        if key in seen:
            continue
        seen.add(key)
        cells.append(clean)
    return cells


def _viewbox_mask(vb: List[float]) -> Polygon:
    x0, y0, w, h = vb
    return Polygon([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)])


def _mask_from_vertices(vb: List[float], vertices: list[list[float]] | list[tuple[float, float]] | None):
    if not vertices or len(vertices) < 3:
        return None
    try:
        poly = Polygon([(float(x), float(y)) for x, y in vertices])
    except Exception:
        return None
    poly = _coerce_polygon(poly, allow_largest=True)
    if poly is None:
        return None
    try:
        clipped = poly.intersection(_viewbox_mask(vb))
    except Exception:
        clipped = poly
    return _clean_polygon(_coerce_polygon(clipped, allow_largest=True), round_snap=0.25)


def _round_weld_boundary_mask(
    vb: List[float],
    vertices: list[list[float]] | list[tuple[float, float]] | None,
    *,
    grid: float = 1.0,
) -> Polygon | None:
    if not vertices or len(vertices) < 3 or grid <= 0:
        return _mask_from_vertices(vb, vertices)
    pts = [(float(x), float(y)) for x, y in vertices]
    snapped: list[tuple[float, float]] = []
    for x, y in pts:
        sx = round(x / grid) * grid
        sy = round(y / grid) * grid
        if snapped and abs(snapped[-1][0] - sx) <= 1e-6 and abs(snapped[-1][1] - sy) <= 1e-6:
            continue
        snapped.append((float(sx), float(sy)))
    if len(snapped) >= 2 and abs(snapped[0][0] - snapped[-1][0]) <= 1e-6 and abs(snapped[0][1] - snapped[-1][1]) <= 1e-6:
        snapped.pop()
    cleaned = _clean_poly_vertices(snapped, min_seg=max(0.5, grid * 0.25), round_snap=grid)
    return _mask_from_vertices(vb, cleaned)


def build_source_voronoi(source_path: Path, count: int | None = None, relax: int = 2, seed: int = 7) -> dict:
    source_count = max(1, int(count or config.TARGET_ZONES))
    vb, _ = _parse_source(source_path.read_bytes())
    _mask_vb, mask = _build_source_mask(source_path)
    source_lines, _ = _load_source_segments(source_path)
    boundary_pts = [(float(x), float(y)) for x, y in _poly_to_vertices(mask)]
    source_seed_target = max(1, min(source_count, 100))
    source_candidates = _collect_source_seed_points(source_lines, mask)
    selected_source_pts = _farthest_point_sample(source_candidates, source_seed_target, mask)
    seed_pts = selected_source_pts + boundary_pts
    if len(seed_pts) < 4:
        rng = random.Random(seed)
        pts = _random_points_in_polygon(mask, max(source_count, 4), rng)
        seed_pts = [tuple(map(float, pt)) for pt in pts]
    vor = Voronoi(np.asarray(seed_pts, dtype=float))
    regions, vertices = _voronoi_finite_polygons_2d(vor)
    cells = []
    for region in regions:
        poly = Polygon(vertices[region]).intersection(mask)
        if not poly.is_empty and poly.area > 1e-6:
            poly = _clean_polygon(_coerce_polygon(poly, allow_largest=True))
            if poly:
                cells.append(poly)
    regularized_cells = _regularize_voronoi_cells(cells, mask)
    snapped_cells = _filter_region_seed_cells(regularized_cells, mask)
    if not snapped_cells:
        snapped_cells = _filter_region_seed_cells(cells, mask)
    if not snapped_cells:
        snapped_cells = regularized_cells or cells
    graph_vertices, graph_segments = _graph_from_cells(snapped_cells)
    graph_vertices, graph_segments = _collapse_tiny_graph_cells(
        snapped_cells,
        graph_vertices,
        graph_segments,
    )
    graph_vertices, graph_segments = _collapse_short_graph_edges(graph_vertices, graph_segments, merge_dist=0.9)
    graph_cells = _cells_from_graph(mask, graph_vertices, graph_segments)
    if graph_cells:
        snapped_cells = graph_cells

    def poly_to_vertices(poly: Polygon):
        return [[float(x), float(y)] for x, y in _poly_to_vertices(poly)]
    return {
        "version": SOURCE_VORONOI_VERSION,
        "viewBox": [float(v) for v in vb],
        "mask": poly_to_vertices(mask),
        "cells": [poly_to_vertices(poly) for poly in snapped_cells],
        "snapped_cells": [poly_to_vertices(poly) for poly in snapped_cells],
        "graph_vertices": graph_vertices,
        "graph_segments": graph_segments,
        "count": len(snapped_cells),
    }


def _assign_regions_to_snapped_cells(region_polys: list[Polygon], snapped_polys: list[Polygon]) -> dict[str, list[int]]:
    snap_region_map: dict[str, list[int]] = {str(i): [] for i in range(len(snapped_polys))}
    snap_centroids = [poly.representative_point() for poly in snapped_polys]
    snap_bounds = [poly.bounds for poly in snapped_polys]

    def _bbox_dist2(a, b) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        dx = 0.0 if ax1 >= bx0 and bx1 >= ax0 else (bx0 - ax1 if ax1 < bx0 else ax0 - bx1)
        dy = 0.0 if ay1 >= by0 and by1 >= ay0 else (by0 - ay1 if ay1 < by0 else ay0 - by1)
        return dx * dx + dy * dy

    for rid, region_poly in enumerate(region_polys):
        if region_poly is None or region_poly.is_empty: continue
        rp = region_poly.representative_point()
        assigned = None
        for zid, snap_poly in enumerate(snapped_polys):
            if snap_poly.covers(rp): assigned = zid; break
        if assigned is None:
            best_overlap = -1.0
            for zid, snap_poly in enumerate(snapped_polys):
                try: overlap = region_poly.intersection(snap_poly).area
                except Exception: overlap = 0.0
                if overlap > best_overlap: best_overlap = overlap; assigned = zid
            if best_overlap <= 1e-6:
                assigned = None
        if assigned is None and snap_centroids:
            region_bounds = region_poly.bounds
            best_dist = float("inf")
            for zid, pt in enumerate(snap_centroids):
                bbox_d2 = _bbox_dist2(region_bounds, snap_bounds[zid])
                if bbox_d2 > 64.0:
                    continue
                d = (float(pt.x)-float(rp.x))**2 + (float(pt.y)-float(rp.y))**2
                if d < best_dist: best_dist = d; assigned = zid
        if assigned is not None:
            snap_region_map[str(int(assigned))].append(int(rid))
    return snap_region_map


def _is_on_boundary(p1: Tuple[float, float], p2: Tuple[float, float], vb: List[float], eps: float = 2.0) -> bool:
    x0, y0, w, h = vb
    x1, y1 = p1; x2, y2 = p2
    if abs(x1 - x0) < eps and abs(x2 - x0) < eps: return True
    if abs(x1 - (x0 + w)) < eps and abs(x2 - (x0 + w)) < eps: return True
    if abs(y1 - y0) < eps and abs(y2 - y0) < eps: return True
    if abs(y1 - (y0 + h)) < eps and abs(y2 - (y0 + h)) < eps: return True
    return False


def _load_source_segments(svg_path: Path) -> tuple[list[LineString], tuple[int, int]]:
    root = ET.parse(svg_path).getroot()
    vb = _parse_viewbox(root)
    canvas = (float(vb[2]), float(vb[3]))
    parent_map = {c: p for p in root.iter() for c in p}
    lines: list[LineString] = []
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag in {"defs", "clipPath"}: continue
        m = svg_utils._get_element_transform(elem, parent_map)
        
        # IGNORE 'rect' as it is often the background or image frame
        if tag == "line":
            try:
                p1 = (float(elem.get("x1", "0")), float(elem.get("y1", "0")))
                p2 = (float(elem.get("x2", "0")), float(elem.get("y2", "0")))
                pts = svg_utils._apply_matrix([p1, p2], m)
                if not _is_on_boundary(pts[0], pts[1], vb): lines.append(LineString(pts))
            except Exception: pass
        elif tag in {"polyline", "polygon"}:
            pts_raw = _parse_points_attr(elem.get("points", ""))
            if len(pts_raw) >= 2:
                pts = svg_utils._apply_matrix(pts_raw, m)
                if tag == "polygon" and pts[0] != pts[-1]: pts.append(pts[0])
                for i in range(len(pts)-1):
                    if not _is_on_boundary(pts[i], pts[i+1], vb): lines.append(LineString([pts[i], pts[i+1]]))
        elif tag == "path":
            for pts_raw in _parse_path_d(elem.get("d", "")):
                if len(pts_raw) >= 2:
                    pts = svg_utils._apply_matrix(pts_raw, m)
                    for i in range(len(pts)-1):
                        if not _is_on_boundary(pts[i], pts[i+1], vb): lines.append(LineString([pts[i], pts[i+1]]))
    return lines, canvas


def _load_cached_source_segments(
    cached_nodes: list[dict] | None,
    cached_segments: list[list[int]] | None,
    vb: list[float],
) -> list[LineString]:
    if not isinstance(cached_nodes, list) or not isinstance(cached_segments, list):
        return []
    nodes_by_id: dict[int, tuple[float, float]] = {}
    for idx, node in enumerate(cached_nodes):
        if not isinstance(node, dict):
            continue
        nid_raw = node.get("id", idx)
        try:
            nid = int(nid_raw)
        except Exception:
            continue
        try:
            x = float(node.get("x", 0.0))
            y = float(node.get("y", 0.0))
        except Exception:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        nodes_by_id[nid] = (x, y)
    if not nodes_by_id:
        return []

    lines: list[LineString] = []
    seen: set[tuple[int, int]] = set()
    for seg in cached_segments:
        if not isinstance(seg, (list, tuple)) or len(seg) < 2:
            continue
        try:
            a_id = int(seg[0])
            b_id = int(seg[1])
        except Exception:
            continue
        if a_id == b_id:
            continue
        a = nodes_by_id.get(a_id)
        b = nodes_by_id.get(b_id)
        if a is None or b is None:
            continue
        if math.hypot(a[0] - b[0], a[1] - b[1]) <= 1e-9:
            continue
        if _is_on_boundary(a, b, vb):
            continue
        edge_key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
        if edge_key in seen:
            continue
        seen.add(edge_key)
        try:
            lines.append(LineString([a, b]))
        except Exception:
            continue
    return lines


def _build_source_mask(svg_path: Path):
    root = ET.parse(svg_path).getroot()
    vb = _parse_viewbox(root)
    fallback = _viewbox_mask(vb)
    try:
        source_lines, _ = _load_source_segments(svg_path)
        if not source_lines:
            return vb, fallback
        merged = unary_union(source_lines)
        poly_pts, _cuts, _dangles, _invalid = polygonize_full(merged)
        geoms = list(poly_pts.geoms) if hasattr(poly_pts, "geoms") else [poly_pts]
        polys = []
        for geom in geoms:
            if geom.is_empty or geom.geom_type != "Polygon" or geom.area <= 1e-6:
                continue
            clipped = geom.intersection(fallback)
            parts = list(clipped.geoms) if hasattr(clipped, "geoms") and clipped.geom_type == "MultiPolygon" else [clipped]
            for part in parts:
                poly = _coerce_polygon(part, allow_largest=True)
                if poly is not None:
                    polys.append(poly)
        if not polys:
            return vb, fallback
        unioned = unary_union(polys)
        mask = _clean_polygon(
            _coerce_polygon(unioned, allow_largest=True),
            simplify_tol=1.1,
            min_seg=1.5,
            round_snap=0.25,
        )
        return vb, mask or fallback
    except Exception:
        return vb, fallback


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
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    return (ax + t * dx, ay + t * dy)


def _snap_boundary_cells_to_source(
    cells: list[Polygon],
    source_lines: list[LineString],
    mask: Polygon,
    *,
    snap_dist: float = 3.25,
    boundary_band: float = 4.5,
) -> list[Polygon]:
    if not cells or not source_lines or mask.is_empty:
        return cells
    segs: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float, float, float]]] = []
    for ln in source_lines:
        coords = list(ln.coords)
        if len(coords) < 2:
            continue
        a = (float(coords[0][0]), float(coords[0][1]))
        b = (float(coords[-1][0]), float(coords[-1][1]))
        minx = min(a[0], b[0]) - snap_dist
        miny = min(a[1], b[1]) - snap_dist
        maxx = max(a[0], b[0]) + snap_dist
        maxy = max(a[1], b[1]) + snap_dist
        segs.append((a, b, (minx, miny, maxx, maxy)))
    out: list[Polygon] = []
    for poly in cells:
        pts = _poly_to_vertices(poly)
        next_pts: list[tuple[float, float]] = []
        for x, y in pts:
            p = (float(x), float(y))
            try:
                near_boundary = Point(p).distance(mask.exterior) <= boundary_band
            except Exception:
                near_boundary = False
            if not near_boundary:
                next_pts.append(p)
                continue
            best = p
            best_d2 = snap_dist * snap_dist
            for a, b, (minx, miny, maxx, maxy) in segs:
                if p[0] < minx or p[0] > maxx or p[1] < miny or p[1] > maxy:
                    continue
                proj = _project_point_to_segment(p, a, b)
                if proj is None:
                    continue
                dx = proj[0] - p[0]
                dy = proj[1] - p[1]
                d2 = dx * dx + dy * dy
                if d2 <= best_d2:
                    best_d2 = d2
                    best = proj
            next_pts.append(best)
        snapped = _clean_polygon(_coerce_polygon(Polygon(_clean_poly_vertices(next_pts)), allow_largest=True))
        out.append(snapped or poly)
    return out


def _regularize_voronoi_cells(
    cells: list[Polygon],
    mask: Polygon,
    *,
    weld_dist: float = 1.8,
) -> list[Polygon]:
    if not cells or mask is None or mask.is_empty:
        return cells
    all_lines: list[LineString] = []
    for poly in cells:
        all_lines.extend(_polygon_boundary_lines(poly))
    all_lines.extend(_polygon_boundary_lines(mask))
    merged = unary_union(_weld_segments(all_lines, weld_dist=weld_dist))
    poly_pts, _cuts, _dangles, _invalid = polygonize_full(merged)
    geoms = list(poly_pts.geoms) if hasattr(poly_pts, "geoms") else [poly_pts]
    polys: list[Polygon] = []
    seen: set[tuple[float, float, float, float]] = set()
    for geom in geoms:
        if geom.is_empty or getattr(geom, "geom_type", "") != "Polygon":
            continue
        clipped = geom.intersection(mask)
        parts = list(clipped.geoms) if hasattr(clipped, "geoms") and clipped.geom_type == "MultiPolygon" else [clipped]
        for part in parts:
            poly = _clean_polygon(_coerce_polygon(part, allow_largest=True), simplify_tol=0.45, min_seg=0.9)
            if poly is None or poly.area <= 1e-4:
                continue
            rep = poly.representative_point()
            try:
                if not mask.buffer(1e-6).covers(rep):
                    continue
            except Exception:
                pass
            key = tuple(round(v, 3) for v in poly.bounds)
            if key in seen:
                continue
            seen.add(key)
            polys.append(poly)
    if not polys:
        return cells
    try:
        covered = unary_union(polys)
        missing = mask.difference(covered)
        extras = list(missing.geoms) if hasattr(missing, "geoms") and missing.geom_type == "MultiPolygon" else [missing]
        for extra in extras:
            poly = _clean_polygon(_coerce_polygon(extra, allow_largest=True), simplify_tol=0.45, min_seg=0.9)
            if poly is not None and poly.area > 1e-3:
                polys.append(poly)
    except Exception:
        pass
    return sorted(polys, key=lambda p: float(p.area), reverse=True)


def _filter_region_seed_cells(
    cells: list[Polygon],
    mask: Polygon,
    *,
    vertex_near_boundary: float = 9.0,
) -> list[Polygon]:
    if not cells or mask is None or mask.is_empty:
        return cells
    kept: list[Polygon] = []
    for poly in cells:
        clean = _clean_polygon(_coerce_polygon(poly, allow_largest=True))
        if clean is None or clean.area <= 1e-6:
            continue
        try:
            if clean.boundary.intersection(mask.boundary).length > 1e-3 or clean.distance(mask.boundary) < 1e-6:
                continue
        except Exception:
            pass
        too_near = False
        for x, y in _poly_to_vertices(clean):
            try:
                if Point(float(x), float(y)).distance(mask.boundary) < vertex_near_boundary:
                    too_near = True
                    break
            except Exception:
                continue
        if not too_near:
            kept.append(clean)
    return kept


def _nearest_perpendicular_drop(
    pt: tuple[float, float],
    cell_segments: list[tuple[int, tuple[float, float], tuple[float, float], float]],
):
    px, py = float(pt[0]), float(pt[1])
    best = None
    best_d2 = float("inf")
    for cell_idx, a, b, ll in cell_segments:
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        if ll <= 1e-9:
            continue
        t = ((px - ax) * dx + (py - ay) * dy) / ll
        if t < 0.0 or t > 1.0:
            continue
        proj = (ax + t * dx, ay + t * dy)
        d2 = (proj[0] - px) ** 2 + (proj[1] - py) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = (cell_idx, proj, math.sqrt(d2))
    return best


def _nearest_cell_vertex(
    pt: tuple[float, float],
    cell_vertices: list[list[tuple[float, float]]],
):
    px, py = float(pt[0]), float(pt[1])
    best = None
    best_d2 = float("inf")
    for cell_idx, verts in enumerate(cell_vertices):
        for vx, vy in verts:
            d2 = (float(vx) - px) ** 2 + (float(vy) - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = (cell_idx, (float(vx), float(vy)), math.sqrt(d2))
    return best


def _segment_intersects_existing(seg: LineString, accepted: list[LineString]) -> bool:
    for other in accepted:
        inter = seg.intersection(other)
        if inter.is_empty:
            continue
        if inter.geom_type == "Point":
            coords = (round(float(inter.x), 6), round(float(inter.y), 6))
            ends = {
                (round(float(seg.coords[0][0]), 6), round(float(seg.coords[0][1]), 6)),
                (round(float(seg.coords[-1][0]), 6), round(float(seg.coords[-1][1]), 6)),
                (round(float(other.coords[0][0]), 6), round(float(other.coords[0][1]), 6)),
                (round(float(other.coords[-1][0]), 6), round(float(other.coords[-1][1]), 6)),
            }
            if coords in ends:
                continue
        return True
    return False


def _segment_crosses_forbidden(
    seg: LineString,
    forbidden: list[LineString],
    *,
    allowed_points: set[tuple[float, float]] | None = None,
) -> bool:
    allowed = allowed_points or set()
    for other in forbidden:
        inter = seg.intersection(other)
        if inter.is_empty:
            continue
        if inter.geom_type == "Point":
            coords = (round(float(inter.x), 6), round(float(inter.y), 6))
            if coords in allowed:
                continue
        return True
    return False


def _build_boundary_connection_lines(
    mask: Polygon,
    cells: list[Polygon],
) -> list[LineString]:
    if mask is None or mask.is_empty or not cells:
        return []
    boundary_pts = _poly_to_vertices(mask)
    boundary_lines = _polygon_boundary_lines(mask)
    cell_vertices: list[tuple[float, float]] = []
    cell_lines: list[LineString] = []
    for poly in cells:
        verts = [(float(x), float(y)) for x, y in _poly_to_vertices(poly)]
        cell_vertices.extend(verts)
        cell_lines.extend(_polygon_boundary_lines(poly))
    cell_vertices = _dedup_xy_points(cell_vertices, digits=3)
    forbidden = cell_lines + boundary_lines
    accepted: list[LineString] = []
    for pt in boundary_pts:
        src = (float(pt[0]), float(pt[1]))
        candidates = sorted(
            cell_vertices,
            key=lambda v: (v[0] - src[0]) ** 2 + (v[1] - src[1]) ** 2,
        )
        for end_pt in candidates:
            seg = LineString([src, (float(end_pt[0]), float(end_pt[1]))])
            if seg.length <= 1e-6:
                continue
            allowed = {
                (round(src[0], 6), round(src[1], 6)),
                (round(float(end_pt[0]), 6), round(float(end_pt[1]), 6)),
            }
            if _segment_crosses_forbidden(seg, forbidden, allowed_points=allowed):
                continue
            if _segment_intersects_existing(seg, accepted):
                continue
            accepted.append(seg)
            break
    return accepted


def _rebuild_cells_with_boundary_connections(
    mask: Polygon,
    cells: list[Polygon],
) -> list[Polygon]:
    if mask is None or mask.is_empty or not cells:
        return cells
    all_lines: list[LineString] = []
    for poly in cells:
        all_lines.extend(_polygon_boundary_lines(poly))
    all_lines.extend(_polygon_boundary_lines(mask))
    all_lines.extend(_build_boundary_connection_lines(mask, cells))
    merged = unary_union(_weld_segments(all_lines, weld_dist=1.2))
    poly_pts, _cuts, _dangles, _invalid = polygonize_full(merged)
    geoms = list(poly_pts.geoms) if hasattr(poly_pts, "geoms") else [poly_pts]
    polys: list[Polygon] = []
    seen: set[tuple[float, float, float, float]] = set()
    for geom in geoms:
        if geom.is_empty or getattr(geom, "geom_type", "") != "Polygon":
            continue
        clipped = geom.intersection(mask)
        parts = list(clipped.geoms) if hasattr(clipped, "geoms") and clipped.geom_type == "MultiPolygon" else [clipped]
        for part in parts:
            poly = _clean_polygon(_coerce_polygon(part, allow_largest=True), simplify_tol=0.35, min_seg=0.75)
            if poly is None or poly.area <= 1e-4:
                continue
            try:
                if not mask.buffer(1e-6).covers(poly.representative_point()):
                    continue
            except Exception:
                pass
            key = tuple(round(v, 3) for v in poly.bounds)
            if key in seen:
                continue
            seen.add(key)
            polys.append(poly)
    if polys:
        try:
            covered = unary_union(polys)
            missing = mask.difference(covered)
            extras = list(missing.geoms) if hasattr(missing, "geoms") and missing.geom_type == "MultiPolygon" else [missing]
            for extra in extras:
                poly = _clean_polygon(_coerce_polygon(extra, allow_largest=True), simplify_tol=0.35, min_seg=0.75)
                if poly is None or poly.area <= 1e-3:
                    continue
                key = tuple(round(v, 3) for v in poly.bounds)
                if key in seen:
                    continue
                seen.add(key)
                polys.append(poly)
        except Exception:
            pass
    return polys or cells



def _write_voronoi_debug_svg(
    out_path: Path,
    vb: list[float],
    mask: Polygon,
    cells: list[Polygon],
) -> Path:
    def _cluster_points(
        pts: list[tuple[float, float]],
        *,
        radius: float = 5.5,
        boundary_snap: float = 4.5,
    ) -> tuple[list[tuple[float, float]], list[int]]:
        if not pts:
            return [], []
        r2 = radius * radius
        cell = max(1.0, radius)
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        groups: list[list[float]] = []
        counts: list[int] = []
        assignments: list[int] = []

        def _nearest_mask_vertex(x: float, y: float) -> tuple[float, float] | None:
            if mask is None or mask.is_empty:
                return None
            best = None
            best_d2 = boundary_snap * boundary_snap
            for mx, my in _poly_to_vertices(mask):
                dx = float(mx) - x
                dy = float(my) - y
                d2 = dx * dx + dy * dy
                if d2 <= best_d2:
                    best_d2 = d2
                    best = (float(mx), float(my))
            return best

        for px, py in pts:
            x = float(px)
            y = float(py)
            snapped = _nearest_mask_vertex(x, y)
            if snapped is not None:
                x, y = snapped
            gx = int(math.floor(x / cell))
            gy = int(math.floor(y / cell))
            best_idx = -1
            best_d2 = r2
            for ix in range(gx - 1, gx + 2):
                for iy in range(gy - 1, gy + 2):
                    for idx in buckets.get((ix, iy), []):
                        cx, cy = groups[idx]
                        dx = x - cx
                        dy = y - cy
                        d2 = dx * dx + dy * dy
                        if d2 <= best_d2:
                            best_d2 = d2
                            best_idx = idx
            if best_idx >= 0:
                cnt = counts[best_idx] + 1
                groups[best_idx][0] = (groups[best_idx][0] * counts[best_idx] + x) / cnt
                groups[best_idx][1] = (groups[best_idx][1] * counts[best_idx] + y) / cnt
                counts[best_idx] = cnt
                assignments.append(best_idx)
                continue
            idx = len(groups)
            groups.append([x, y])
            counts.append(1)
            buckets[(gx, gy)].append(idx)
            assignments.append(idx)
        return [(float(x), float(y)) for x, y in groups], assignments

    x0, y0, w, h = [float(v) for v in vb]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="{x0} {y0} {w} {h}">',
        '<rect width="100%" height="100%" fill="#081224"/>',
    ]
    if mask is not None and not mask.is_empty:
        mask_pts = " ".join(f"{x:.3f},{y:.3f}" for x, y in _poly_to_vertices(mask))
        parts.append(f'<polygon points="{mask_pts}" fill="rgba(0,210,106,0.08)" stroke="#2f80ff" stroke-width="2"/>')
    boundary_pts: list[tuple[float, float]] = []
    raw_edges: list[tuple[int, int]] = []
    for poly in cells:
        pts = _poly_to_vertices(poly)
        n = len(pts)
        if n < 2:
            continue
        base = len(boundary_pts)
        for x, y in pts:
            boundary_pts.append((float(x), float(y)))
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            try:
                mid = Point((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
                is_boundary_edge = mid.distance(mask.boundary) <= 2.25
            except Exception:
                is_boundary_edge = False
            if not is_boundary_edge:
                continue
            raw_edges.append((base + i, base + ((i + 1) % n)))
    clustered_pts, assignments = _cluster_points(boundary_pts, radius=6.0, boundary_snap=5.5)
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for a_raw, b_raw in raw_edges:
        if a_raw >= len(assignments) or b_raw >= len(assignments):
            continue
        a = assignments[a_raw]
        b = assignments[b_raw]
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        edge_counts[key] += 1
        adjacency[a].add(b)
        adjacency[b].add(a)
    suspicious_nodes: set[int] = set()
    suspicious_edges: set[tuple[int, int]] = set()
    for key, count in edge_counts.items():
        a, b = key
        ax, ay = clustered_pts[a]
        bx, by = clustered_pts[b]
        seg_len = math.hypot(bx - ax, by - ay)
        if count > 1 or seg_len < 10.0:
            suspicious_edges.add(key)
            suspicious_nodes.add(a)
            suspicious_nodes.add(b)
    for idx, nbrs in adjacency.items():
        if len(nbrs) != 2:
            suspicious_nodes.add(idx)
            for j in nbrs:
                suspicious_edges.add((idx, j) if idx < j else (j, idx))
    for (a, b), count in sorted(edge_counts.items()):
        ax, ay = clustered_pts[a]
        bx, by = clustered_pts[b]
        is_bad = (a, b) in suspicious_edges
        color = "#ff4d4f" if is_bad else "#7CFFB2"
        width = "2.4" if is_bad else "1.1"
        opacity = "1.0" if is_bad else "0.65"
        parts.append(
            f'<line x1="{ax:.3f}" y1="{ay:.3f}" x2="{bx:.3f}" y2="{by:.3f}" stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>'
        )
        if count > 1:
            mx = (ax + bx) * 0.5
            my = (ay + by) * 0.5
            parts.append(
                f'<text x="{mx:.3f}" y="{my:.3f}" fill="#ffd400" font-size="6" text-anchor="middle">{count}</text>'
            )
    for idx, (x, y) in enumerate(clustered_pts):
        deg = len(adjacency.get(idx, set()))
        is_bad = idx in suspicious_nodes
        fill = "#ff4d4f" if is_bad else "#7CFFB2"
        parts.append(
            f'<circle cx="{float(x):.3f}" cy="{float(y):.3f}" r="2.8" fill="{fill}" stroke="#ffffff" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{float(x) + 4:.3f}" y="{float(y) - 4:.3f}" fill="#ffffff" font-size="6">{idx}:{deg}</text>'
        )
    parts.append("</svg>")
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path


def normalize_source_voronoi_payload(source_path: Path, payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return build_source_voronoi(source_path)
    root = ET.parse(source_path).getroot()
    vb = payload.get("viewBox") or _parse_viewbox(root)
    payload_mask = payload.get("mask")
    mask = _round_weld_boundary_mask(vb, payload_mask, grid=1.0) if payload_mask else None
    mask = mask or _mask_from_vertices(vb, payload_mask) or _build_source_mask(source_path)[1]
    raw_graph_vertices = payload.get("graph_vertices") or payload.get("graphVertices") or []
    raw_graph_segments = payload.get("graph_segments") or payload.get("graphSegments") or []
    if raw_graph_vertices and raw_graph_segments:
        graph_vertices: list[list[float]] = []
        for pt in raw_graph_vertices:
            if not isinstance(pt, list) or len(pt) < 2:
                continue
            graph_vertices.append([float(pt[0]), float(pt[1])])
        graph_segments: list[list[int]] = []
        for seg in raw_graph_segments:
            if not isinstance(seg, list) or len(seg) < 2:
                continue
            a = int(seg[0])
            b = int(seg[1])
            if a == b:
                continue
            graph_segments.append([a, b])
        base_cells = _cells_from_graph(mask, graph_vertices, graph_segments)
        graph_vertices, graph_segments = _collapse_tiny_graph_cells(
            base_cells,
            graph_vertices,
            graph_segments,
        )
        graph_vertices, graph_segments = _collapse_short_graph_edges(graph_vertices, graph_segments, merge_dist=0.9)
        graph_vertices, graph_segments = _round_weld_graph(graph_vertices, graph_segments, grid=1.0)
        final_cells = _cells_from_graph(mask, graph_vertices, graph_segments)
        if final_cells:
            out_path = source_path.parent.parent / "scripts" / f"tmp_voronoi_regularized_{source_path.stem}.svg"
            try:
                _write_voronoi_debug_svg(out_path, vb, mask, final_cells)
            except Exception:
                pass

            def poly_to_vertices(poly: Polygon):
                return [[float(x), float(y)] for x, y in _poly_to_vertices(poly)]

            cell_vertices = [poly_to_vertices(poly) for poly in final_cells]
            return {
                "version": SOURCE_VORONOI_VERSION,
                "viewBox": [float(v) for v in vb],
                "mask": poly_to_vertices(mask),
                "cells": cell_vertices,
                "snapped_cells": cell_vertices,
                "snappedCells": cell_vertices,
                "graph_vertices": graph_vertices,
                "graph_segments": graph_segments,
                "count": len(cell_vertices),
            }
    raw_cells = (
        payload.get("snapped_cells")
        or payload.get("snappedCells")
        or payload.get("cells")
        or []
    )
    cells: list[Polygon] = []
    for pts in raw_cells:
        if not isinstance(pts, list) or len(pts) < 3:
            continue
        poly = _clean_polygon(_coerce_polygon(Polygon(pts), allow_largest=True), simplify_tol=0.45, min_seg=0.85)
        if poly is not None and poly.area > 1e-4:
            cells.append(poly)
    if not cells:
        return build_source_voronoi(
            source_path,
            count=payload.get("count") if isinstance(payload.get("count"), int) else None,
        )
    payload_version = int(payload.get("version") or 0)
    # When cells come from the frontend editor, preserve the user's manual Voronoi graph.
    # Only apply light cleanup instead of rebuilding topology from scratch.
    if payload_version == SOURCE_VORONOI_VERSION:
        final_cells = []
        for poly in cells:
            clean = _clean_polygon(poly, simplify_tol=0.18, min_seg=0.45)
            if clean is not None and clean.area > 1e-4:
                final_cells.append(clean)
        final_cells = final_cells or cells
    else:
        regularized = _regularize_voronoi_cells(cells, mask)
        final_cells = _filter_region_seed_cells(regularized or cells, mask) or regularized or cells

    def poly_to_vertices(poly: Polygon):
        return [[float(x), float(y)] for x, y in _poly_to_vertices(poly)]

    graph_vertices, graph_segments = _graph_from_cells(final_cells)
    graph_vertices, graph_segments = _collapse_tiny_graph_cells(
        final_cells,
        graph_vertices,
        graph_segments,
    )
    graph_vertices, graph_segments = _collapse_short_graph_edges(graph_vertices, graph_segments, merge_dist=0.9)
    graph_vertices, graph_segments = _round_weld_graph(graph_vertices, graph_segments, grid=1.0)
    graph_cells = _cells_from_graph(mask, graph_vertices, graph_segments)
    if graph_cells:
        final_cells = graph_cells
    out_path = source_path.parent.parent / "scripts" / f"tmp_voronoi_regularized_{source_path.stem}.svg"
    try:
        _write_voronoi_debug_svg(out_path, vb, mask, final_cells)
    except Exception:
        pass
    cell_vertices = [poly_to_vertices(poly) for poly in final_cells]
    return {
        "version": SOURCE_VORONOI_VERSION,
        "viewBox": [float(v) for v in vb],
        "mask": poly_to_vertices(mask),
        "cells": cell_vertices,
        "snapped_cells": cell_vertices,
        "snappedCells": cell_vertices,
        "graph_vertices": graph_vertices,
        "graph_segments": graph_segments,
        "count": len(cell_vertices),
    }


def _weld_segments(
    lines: list[LineString],
    *,
    weld_dist: float = 0.9,
) -> list[LineString]:
    if not lines:
        return lines
    weld_d2 = weld_dist * weld_dist
    cell = max(0.25, weld_dist)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    reps: list[list[float]] = []
    counts: list[int] = []

    def _find_or_add(pt: tuple[float, float]) -> int:
        x = float(pt[0])
        y = float(pt[1])
        gx = int(math.floor(x / cell))
        gy = int(math.floor(y / cell))
        best_idx = -1
        best_d2 = weld_d2
        for ix in range(gx - 1, gx + 2):
            for iy in range(gy - 1, gy + 2):
                for idx in buckets.get((ix, iy), []):
                    rx, ry = reps[idx]
                    dx = x - rx
                    dy = y - ry
                    d2 = dx * dx + dy * dy
                    if d2 <= best_d2:
                        best_d2 = d2
                        best_idx = idx
        if best_idx >= 0:
            cnt = counts[best_idx] + 1
            reps[best_idx][0] = (reps[best_idx][0] * counts[best_idx] + x) / cnt
            reps[best_idx][1] = (reps[best_idx][1] * counts[best_idx] + y) / cnt
            counts[best_idx] = cnt
            return best_idx
        idx = len(reps)
        reps.append([x, y])
        counts.append(1)
        buckets[(gx, gy)].append(idx)
        return idx

    out: list[LineString] = []
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for ln in lines:
        coords = list(ln.coords)
        if len(coords) < 2:
            continue
        a_idx = _find_or_add((coords[0][0], coords[0][1]))
        b_idx = _find_or_add((coords[-1][0], coords[-1][1]))
        ax, ay = reps[a_idx]
        bx, by = reps[b_idx]
        if (ax - bx) * (ax - bx) + (ay - by) * (ay - by) <= 1e-8:
            continue
        a = (round(ax, 4), round(ay, 4))
        b = (round(bx, 4), round(by, 4))
        key = (a, b) if a <= b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append(LineString([a, b]))
    return out


def _keep_region_fast(poly: Polygon) -> bool:
    if poly is None or poly.is_empty:
        return False
    minx, miny, maxx, maxy = poly.bounds
    bw = max(0.0, float(maxx) - float(minx))
    bh = max(0.0, float(maxy) - float(miny))
    if bw <= 0.0 or bh <= 0.0:
        return False
    bbox_area = bw * bh
    short_side = min(bw, bh)
    long_side = max(bw, bh)
    if bbox_area < 2.5:
        return False
    if short_side < 0.2:
        return False
    if short_side < 0.75 and long_side < 8.0:
        return False
    if short_side < 0.5 and long_side < 16.0:
        return False
    return True


def _point_seg_dist(pt: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    proj = _project_point_to_segment(pt, a, b)
    if proj is None:
        return float("inf")
    dx = float(pt[0]) - float(proj[0])
    dy = float(pt[1]) - float(proj[1])
    return math.hypot(dx, dy)


def _prune_region_spikes(
    pts: list[tuple[float, float]],
    *,
    spike_height: float = 0.65,
    tail_width: float = 0.85,
    tail_area2: float = 1.8,
    max_passes: int = 4,
) -> list[tuple[float, float]]:
    if len(pts) < 4:
        return pts
    cur = [(float(x), float(y)) for x, y in pts]
    for _ in range(max_passes):
        if len(cur) < 4:
            break
        changed = False
        nxt: list[tuple[float, float]] = []
        n = len(cur)
        for i in range(n):
            a = cur[(i - 1) % n]
            b = cur[i]
            c = cur[(i + 1) % n]
            ab = math.hypot(b[0] - a[0], b[1] - a[1])
            bc = math.hypot(c[0] - b[0], c[1] - b[1])
            ac = math.hypot(c[0] - a[0], c[1] - a[1])
            tri2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            height = _point_seg_dist(b, a, c)
            # Remove tiny needle/spike vertices that only create a very thin protrusion.
            if (
                height <= spike_height
                and min(ab, bc) <= tail_width
                and tri2 <= tail_area2
                and ac <= max(ab, bc) + 1.25
            ):
                changed = True
                continue
            nxt.append(b)
        if not changed or len(nxt) == len(cur):
            cur = nxt
            break
        cur = nxt
    return cur if len(cur) >= 3 else []


def _sanitize_region_poly(poly: Polygon | None) -> Polygon | None:
    clean = _coerce_polygon(poly, allow_largest=True)
    if clean is None or clean.is_empty:
        return None
    # Keep region topology close to the polygonized source+voronoi graph.
    # Over-aggressive cleanup removes thin but valid closed faces near the boundary.
    pts = _clean_poly_vertices(_poly_to_vertices(clean), min_seg=0.12)
    pts = _prune_region_spikes(
        pts,
        spike_height=0.22,
        tail_width=0.28,
        tail_area2=0.25,
        max_passes=2,
    )
    pts = _clean_poly_vertices(pts, min_seg=0.18)
    if len(pts) < 3:
        return None
    return _coerce_polygon(Polygon(pts), allow_largest=True)


def _iter_line_parts(geom) -> Iterable[LineString]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, LineString):
        return [geom]
    if hasattr(geom, "geoms"):
        out: list[LineString] = []
        for g in geom.geoms:
            if isinstance(g, LineString):
                out.append(g)
        return out
    return []


def _poly_vertices(poly: Polygon) -> list[tuple[float, float]]:
    if poly is None or poly.is_empty:
        return []
    try:
        coords = list(poly.exterior.coords)
    except Exception:
        return []
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [(float(x), float(y)) for x, y in coords]


def _poly_strictly_inside_cell(poly: Polygon, cell: Polygon, outside_tol: float = 0.05) -> bool:
    if poly is None or poly.is_empty or cell is None or cell.is_empty:
        return False
    verts = _poly_vertices(poly)
    if len(verts) < 3:
        return False
    try:
        cover = cell.buffer(float(outside_tol))
    except GEOSException:
        cover = cell
    for x, y in verts:
        try:
            if not cover.covers(Point(x, y)):
                return False
        except GEOSException:
            return False
    return True


def _postprocess_cell_regions(
    cell: Polygon,
    polys: list[Polygon],
    boundary_vertex_thresh: float = 5.0,
    outside_tol: float = 0.05,
) -> list[Polygon]:
    if cell is None or cell.is_empty or not polys:
        return []

    prepared: list[Polygon] = []
    for poly in polys:
        clean = _sanitize_region_poly(poly) or _coerce_polygon(poly, allow_largest=True)
        if clean is None or clean.is_empty or float(clean.area) <= 1e-6:
            continue
        if not _poly_strictly_inside_cell(clean, cell, outside_tol=outside_tol):
            continue
        prepared.append(clean)

    if len(prepared) <= 1:
        return prepared

    cell_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    cell_vs = _poly_vertices(cell)
    if len(cell_vs) >= 3:
        for i in range(len(cell_vs)):
            a = cell_vs[i]
            b = cell_vs[(i + 1) % len(cell_vs)]
            if math.hypot(b[0] - a[0], b[1] - a[1]) <= 1e-9:
                continue
            cell_edges.append((a, b))

    def _point_seg_dist(pt: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        px, py = pt
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        den = dx * dx + dy * dy
        if den <= 1e-12:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / den
        t = max(0.0, min(1.0, t))
        qx = ax + t * dx
        qy = ay + t * dy
        return math.hypot(px - qx, py - qy)

    def _matched_cell_edge_idx(poly: Polygon, threshold: float) -> int:
        verts = _poly_vertices(poly)
        if not verts or not cell_edges:
            return -1
        for ei, (a, b) in enumerate(cell_edges):
            ok = True
            for v in verts:
                if _point_seg_dist(v, a, b) >= threshold:
                    ok = False
                    break
            if ok:
                return ei
        return -1

    boundary_like: list[Polygon] = []
    survivors: list[Polygon] = []
    thr = max(0.0, float(boundary_vertex_thresh))
    for poly in prepared:
        edge_idx = _matched_cell_edge_idx(poly, thr)
        if edge_idx >= 0:
            boundary_like.append(poly)
        else:
            survivors.append(poly)

    if not boundary_like:
        return prepared

    for small in sorted(boundary_like, key=lambda p: float(p.area)):
        best_idx = -1
        best_shared = 0.0
        for i, cand in enumerate(survivors):
            try:
                shared = float(small.boundary.intersection(cand.boundary).length)
            except GEOSException:
                shared = 0.0
            if shared > best_shared:
                best_shared = shared
                best_idx = i
        if best_idx < 0 or best_shared <= 1e-6:
            survivors.append(small)
            continue
        try:
            merged = unary_union([survivors[best_idx], small])
        except GEOSException:
            survivors.append(small)
            continue
        merged_poly = _coerce_polygon(merged, allow_largest=True)
        merged_poly = _sanitize_region_poly(merged_poly) if merged_poly is not None else None
        if merged_poly is None or merged_poly.is_empty:
            survivors.append(small)
            continue
        try:
            merged_poly = _coerce_polygon(merged_poly.intersection(cell), allow_largest=True) or merged_poly
        except GEOSException:
            pass
        if not _poly_strictly_inside_cell(merged_poly, cell, outside_tol=outside_tol):
            survivors.append(small)
            continue
        survivors[best_idx] = merged_poly

    return [p for p in survivors if p is not None and not p.is_empty and float(p.area) > 1e-6]


def _snap_source_lines_to_voronoi_vertices(
    source_lines: list[LineString],
    snapped_polys: list[Polygon],
    threshold: float = 1.0,
) -> list[LineString]:
    if not source_lines or not snapped_polys or threshold <= 0:
        return source_lines
    verts: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for poly in snapped_polys:
        if poly is None or poly.is_empty:
            continue
        try:
            coords = list(poly.exterior.coords)
        except Exception:
            continue
        if coords and coords[0] == coords[-1]:
            coords = coords[:-1]
        for x, y in coords:
            key = (round(float(x), 3), round(float(y), 3))
            if key in seen:
                continue
            seen.add(key)
            verts.append((float(x), float(y)))
    if not verts:
        return source_lines

    thr2 = float(threshold) * float(threshold)
    snapped_lines: list[LineString] = []
    for ln in source_lines:
        try:
            coords = [(float(x), float(y)) for x, y in ln.coords]
        except Exception:
            continue
        if len(coords) < 2:
            continue
        next_coords: list[tuple[float, float]] = []
        for x, y in coords:
            best = (x, y)
            best_d2 = thr2
            for vx, vy in verts:
                dx = vx - x
                dy = vy - y
                d2 = dx * dx + dy * dy
                if d2 <= best_d2:
                    best_d2 = d2
                    best = (vx, vy)
            if next_coords:
                px, py = next_coords[-1]
                if math.hypot(best[0] - px, best[1] - py) <= 1e-9:
                    continue
            next_coords.append(best)
        if len(next_coords) < 2:
            next_coords = coords
        snapped_lines.append(LineString(next_coords))
    return snapped_lines


def _build_regions_from_voronoi_cells(
    snapped_polys: list[Polygon],
    source_lines: list[LineString],
) -> tuple[list[list[list[float]]], list[Polygon], dict[str, list[int]]]:
    regions: list[list[list[float]]] = []
    region_polys: list[Polygon] = []
    snap_region_map: dict[str, list[int]] = {str(i): [] for i in range(len(snapped_polys))}
    if not snapped_polys:
        return regions, region_polys, snap_region_map

    source_bounds = [ln.bounds for ln in source_lines]

    def _bbox_hits(a, b) -> bool:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return ax1 >= bx0 and bx1 >= ax0 and ay1 >= by0 and by1 >= ay0

    def _poly_within_cell(poly: Polygon, cell: Polygon) -> bool:
        if poly is None or poly.is_empty:
            return False
        try:
            outside = poly.difference(cell)
        except GEOSException:
            return False
        if outside.is_empty:
            return True
        try:
            return float(outside.area) <= max(1e-4, min(0.01, float(poly.area) * 0.0001))
        except Exception:
            return False

    for zid, cell in enumerate(snapped_polys):
        if cell is None or cell.is_empty:
            continue
        cell_bounds = cell.bounds
        local_lines: list[LineString] = []
        for ln, ln_bounds in zip(source_lines, source_bounds):
            if not _bbox_hits(cell_bounds, ln_bounds):
                continue
            try:
                touches = ln.intersects(cell)
            except GEOSException:
                continue
            if not touches:
                continue
            coords = list(ln.coords)
            if len(coords) < 2:
                continue
            # Preserve the full source polyline whenever it touches the Voronoi cell.
            # We still clip the resulting faces back to the cell after polygonize.
            local_lines.append(LineString([(float(x), float(y)) for x, y in coords]))
        all_lines = _weld_segments(local_lines + _polygon_boundary_lines(cell), weld_dist=0.35)
        if not all_lines:
            clean_cell = _sanitize_region_poly(cell) or _coerce_polygon(cell, allow_largest=True)
            if clean_cell is not None and not clean_cell.is_empty:
                coords = list(clean_cell.exterior.coords)
                if coords and coords[0] == coords[-1]:
                    coords = coords[:-1]
                rid = len(regions)
                regions.append([[float(x), float(y)] for x, y in coords])
                region_polys.append(clean_cell)
                snap_region_map[str(zid)].append(rid)
            continue
        merged = unary_union(all_lines)
        poly_pts, _cuts, _dangles, _invalid = polygonize_full(merged)
        kept_local: list[Polygon] = []
        geoms = poly_pts.geoms if hasattr(poly_pts, "geoms") else [poly_pts]
        for geom in geoms:
            if geom.is_empty or geom.geom_type != "Polygon":
                continue
            try:
                clipped = geom.intersection(cell)
            except GEOSException:
                continue
            if clipped.is_empty or clipped.area <= 1e-6:
                continue
            parts = list(clipped.geoms) if hasattr(clipped, "geoms") and clipped.geom_type == "MultiPolygon" else [clipped]
            for part in parts:
                part_poly = _sanitize_region_poly(part)
                if part_poly is None or part_poly.area <= 1e-6:
                    continue
                if not _poly_within_cell(part_poly, cell):
                    continue
                kept_local.append(part_poly)
        if kept_local:
            try:
                covered = unary_union(kept_local)
                missing = cell.difference(covered)
                extras = list(missing.geoms) if hasattr(missing, "geoms") and missing.geom_type == "MultiPolygon" else [missing]
                for extra in extras:
                    poly = _sanitize_region_poly(extra) or _coerce_polygon(extra, allow_largest=True)
                    if poly is None or poly.is_empty or poly.area <= 1e-4:
                        continue
                    if not _poly_within_cell(poly, cell):
                        continue
                    kept_local.append(poly)
            except GEOSException:
                pass
        else:
            poly = _sanitize_region_poly(cell) or _coerce_polygon(cell, allow_largest=True)
            if poly is not None and not poly.is_empty and poly.area > 1e-6 and _poly_within_cell(poly, cell):
                kept_local.append(poly)

        kept_local = _postprocess_cell_regions(cell, kept_local, boundary_vertex_thresh=5.0, outside_tol=0.05)

        seen_local: set[tuple[tuple[float, float], ...]] = set()
        for poly in kept_local:
            if not _poly_within_cell(poly, cell):
                continue
            coords = list(poly.exterior.coords)
            if coords and coords[0] == coords[-1]:
                coords = coords[:-1]
            key = tuple((round(float(x), 4), round(float(y), 4)) for x, y in coords)
            if len(coords) < 3 or key in seen_local:
                continue
            seen_local.add(key)
            rid = len(regions)
            regions.append([[float(x), float(y)] for x, y in coords])
            region_polys.append(poly)
            snap_region_map[str(zid)].append(rid)

    return regions, region_polys, snap_region_map


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
    voronoi = (
        normalize_source_voronoi_payload(source_path, cached_voronoi)
        if cached_voronoi
        else build_source_voronoi(source_path, count=count, relax=relax, seed=seed)
    )
    vb = voronoi.get("viewBox") or _parse_viewbox(ET.parse(source_path).getroot())
    canvas = (float(vb[2]), float(vb[3]))
    mask = _mask_from_vertices(vb, voronoi.get("mask")) or _build_source_mask(source_path)[1]

    source_lines_svg, _ = _load_source_segments(source_path)
    source_lines_cached = _load_cached_source_segments(cached_nodes, cached_segments, vb)
    source_lines = source_lines_cached or source_lines_svg
    voronoi_lines: list[LineString] = []
    voronoi_cells_raw = voronoi.get("snapped_cells") or voronoi.get("cells") or []
    snapped_polys: list[Polygon] = []
    graph_vertices = voronoi.get("graph_vertices") or voronoi.get("graphVertices") or []
    graph_segments = voronoi.get("graph_segments") or voronoi.get("graphSegments") or []
    if graph_vertices and graph_segments:
        base_cells = snapped_polys or [
            _coerce_polygon(Polygon(pts), allow_largest=True)
            for pts in voronoi_cells_raw
            if isinstance(pts, list) and len(pts) >= 3
        ]
        base_cells = [poly for poly in base_cells if poly is not None]
        graph_vertices, graph_segments = _collapse_tiny_graph_cells(
            base_cells,
            graph_vertices,
            graph_segments,
        )
        graph_vertices, graph_segments = _collapse_short_graph_edges(graph_vertices, graph_segments, merge_dist=0.9)
        voronoi_lines.extend(_graph_lines(graph_vertices, graph_segments))
        rebuilt_polys = _cells_from_graph(mask, graph_vertices, graph_segments)
        if rebuilt_polys:
            snapped_polys.extend(rebuilt_polys)
    if not snapped_polys:
        for pts in voronoi_cells_raw:
            p = _coerce_polygon(Polygon(pts), allow_largest=True)
            if p is not None:
                snapped_polys.append(p)
            if len(pts) >= 3:
                for i in range(len(pts)):
                    p1, p2 = pts[i], pts[(i+1)%len(pts)]
                    if not _is_on_boundary(p1, p2, vb):
                        voronoi_lines.append(LineString([p1, p2]))

    source_lines = _snap_source_lines_to_voronoi_vertices(source_lines, snapped_polys, threshold=5.0)
    regions, region_polys, snap_region_map = _build_regions_from_voronoi_cells(snapped_polys, source_lines)

    clean_regions: list[list[list[float]]] = []
    clean_region_polys: list[Polygon] = []
    rid_remap: dict[int, int] = {}
    for rid, pts in enumerate(regions):
        if not isinstance(pts, list) or len(pts) < 3:
            continue
        poly = _coerce_polygon(Polygon(pts), allow_largest=True)
        poly = _sanitize_region_poly(poly) if poly is not None else None
        if poly is None or poly.is_empty or float(poly.area) <= 1e-5:
            continue
        coords = list(poly.exterior.coords)
        if coords and coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) < 3:
            continue
        next_rid = len(clean_regions)
        rid_remap[rid] = next_rid
        clean_regions.append([[float(x), float(y)] for x, y in coords])
        clean_region_polys.append(poly)

    clean_snap_region_map: dict[str, list[int]] = {str(i): [] for i in range(len(snapped_polys))}
    for zid in range(len(snapped_polys)):
        mapped: list[int] = []
        for rid in snap_region_map.get(str(zid), []):
            nrid = rid_remap.get(int(rid))
            if nrid is None:
                continue
            mapped.append(int(nrid))
        clean_snap_region_map[str(zid)] = mapped

    colors_bgr, _ = geometry.compute_region_colors(
        [[(float(x), float(y)) for x, y in pts] for pts in clean_regions],
        canvas,
        svg_path=source_path,
    )
    raw_colors = [f"#{r:02x}{g:02x}{b:02x}" for (b, g, r) in colors_bgr]
    base_colors = [
        (raw_colors[rid] if rid < len(raw_colors) else "")
        for rid in range(len(clean_regions))
    ]
    region_colors = _repair_region_colors_by_adjacency(clean_region_polys, base_colors)
    
    return {
        "version": SOURCE_VORONOI_VERSION,
        "canvas": {"w": int(canvas[0]), "h": int(canvas[1])},
        "regions": clean_regions,
        "region_colors": region_colors,
        "snap_region_map": clean_snap_region_map,
        "source_name": source_path.name,
        "voronoi": voronoi,
    }
