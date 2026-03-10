from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import cv2
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from . import config


def _clean_polygon(poly: Polygon | BaseGeometry | None) -> Polygon | None:
    if poly is None or poly.is_empty:
        return None
    geom = poly if poly.is_valid else poly.buffer(0)
    if geom.is_empty:
        return None
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area, default=None)
    if geom is None or geom.is_empty:
        return None
    if not isinstance(geom, Polygon):
        hull = geom.convex_hull
        if hull.is_empty or not isinstance(hull, Polygon):
            return None
        geom = hull
    if geom.area <= 1e-6:
        return None
    return geom


def _build_region_adjacency(poly_objs: List[Polygon]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in poly_objs]
    bounds = [p.bounds if not p.is_empty else (0.0, 0.0, 0.0, 0.0) for p in poly_objs]
    for i in range(len(poly_objs)):
        p1 = poly_objs[i]
        if p1.is_empty:
            continue
        x0, y0, x1, y1 = bounds[i]
        for j in range(i + 1, len(poly_objs)):
            p2 = poly_objs[j]
            if p2.is_empty:
                continue
            a0, b0, a1, b1 = bounds[j]
            if a1 < x0 or a0 > x1 or b1 < y0 or b0 > y1:
                continue
            inter = p1.boundary.intersection(p2.boundary)
            if inter.is_empty:
                continue
            if inter.length > config.MIDPOINT_EPS:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _generate_points_in_geom(boundary: BaseGeometry, count: int, rng: random.Random) -> List[Tuple[float, float]]:
    minx, miny, maxx, maxy = boundary.bounds
    pts: List[Tuple[float, float]] = []
    attempts = 0
    max_attempts = max(2000, count * 400)
    while len(pts) < count and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if boundary.covers(Point(x, y)):
            pts.append((float(x), float(y)))
    if len(pts) >= count:
        return pts
    centroids: List[Tuple[float, float]] = []
    if isinstance(boundary, MultiPolygon):
        parts = [p for p in boundary.geoms if not p.is_empty and p.area > 1e-6]
    elif isinstance(boundary, Polygon):
        parts = [boundary]
    else:
        parts = []
    for part in parts:
        c = part.representative_point()
        centroids.append((float(c.x), float(c.y)))
    for pt in centroids:
        if len(pts) >= count:
            break
        pts.append(pt)
    while len(pts) < count:
        if centroids:
            pts.append(centroids[len(pts) % len(centroids)])
        else:
            pts.append((float(minx), float(miny)))
    return pts[:count]


def _voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None) -> Tuple[List[List[int]], np.ndarray]:
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi input must be 2D")
    new_regions: List[List[int]] = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = float(np.ptp(vor.points, axis=0).max() * 2.0)

    all_ridges: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges[p1].append((p2, v1, v2))
        all_ridges[p2].append((p1, v1, v2))

    for point_index, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if not region:
            new_regions.append([])
            continue
        if all(v >= 0 for v in region):
            new_regions.append(region)
            continue

        ridges = all_ridges.get(point_index, [])
        new_region = [v for v in region if v >= 0]
        for neighbor_index, v1, v2 in ridges:
            if v1 >= 0 and v2 >= 0:
                continue
            tangent = vor.points[neighbor_index] - vor.points[point_index]
            norm = np.linalg.norm(tangent)
            if norm == 0:
                continue
            tangent /= norm
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[point_index, neighbor_index]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            finite_vertex = vor.vertices[v1 if v1 >= 0 else v2]
            far_point = finite_vertex + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def _build_voronoi_cells(poly_objs: List[Polygon], target: int) -> List[Polygon]:
    valid_polys = [p for p in poly_objs if not p.is_empty and p.area > 1e-6]
    if not valid_polys:
        return []
    boundary = unary_union(valid_polys)
    boundary = boundary.buffer(0)
    if boundary.is_empty:
        return []
    seed_count = max(1, min(int(target), len(valid_polys)))
    rng = random.Random(42)
    best_cells: List[Polygon] = []
    best_score = float("inf")
    for _ in range(12):
        seeds = _generate_points_in_geom(boundary, seed_count, rng)
        if len(seeds) < 4:
            break
        try:
            vor = Voronoi(np.asarray(seeds, dtype=float))
            regions, vertices = _voronoi_finite_polygons_2d(vor)
        except Exception:
            continue
        cells: List[Polygon] = []
        for region in regions:
            if not region:
                continue
            try:
                clipped = Polygon(vertices[region]).buffer(0).intersection(boundary).buffer(0)
            except Exception:
                continue
            cell = _clean_polygon(clipped)
            if cell is not None:
                cells.append(cell)
        if not cells:
            continue
        areas = [c.area for c in cells if c.area > 1e-6]
        if not areas:
            continue
        avg = float(sum(areas) / len(areas))
        std = float(np.std(areas)) if len(areas) > 1 else 0.0
        score = abs(len(cells) - seed_count) * 500.0 + std + abs(avg * len(cells) - boundary.area) * 0.01
        if score < best_score:
            best_score = score
            best_cells = cells
        if len(cells) == seed_count:
            break
    if best_cells:
        return best_cells
    centroid_cells: List[Polygon] = []
    for poly in valid_polys[:seed_count]:
        rep = poly.representative_point()
        buf = rep.buffer(max(1.0, math.sqrt(max(poly.area, 1.0)) * 0.5))
        cell = _clean_polygon(buf.intersection(boundary))
        if cell is not None:
            centroid_cells.append(cell)
    return centroid_cells


def _assign_regions_to_voronoi_cells(poly_objs: List[Polygon], cells: List[Polygon]) -> List[int]:
    if not poly_objs:
        return []
    if not cells:
        return list(range(len(poly_objs)))
    zone_id = [-1] * len(poly_objs)
    cell_centroids = [cell.representative_point() for cell in cells]
    for rid, poly in enumerate(poly_objs):
        if poly.is_empty:
            continue
        rp = poly.representative_point()
        assigned = None
        for zid, cell in enumerate(cells):
            if cell.covers(rp):
                assigned = zid
                break
        if assigned is None:
            best_overlap = -1.0
            for zid, cell in enumerate(cells):
                try:
                    overlap = poly.intersection(cell).area
                except Exception:
                    overlap = 0.0
                if overlap > best_overlap + 1e-9:
                    best_overlap = overlap
                    assigned = zid
        if assigned is None:
            best_dist = float("inf")
            for zid, pt in enumerate(cell_centroids):
                dx = float(pt.x) - float(rp.x)
                dy = float(pt.y) - float(rp.y)
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    assigned = zid
        zone_id[rid] = int(assigned if assigned is not None else rid)
    return zone_id


def build_zones(polys: List[List[Tuple[float, float]]], target: int) -> List[int]:
    if not polys:
        return []

    poly_objs = [_clean_polygon(Polygon(p)) or Polygon() for p in polys]
    adj = _build_region_adjacency(poly_objs)
    voronoi_cells = _build_voronoi_cells(poly_objs, target)
    zone_id = _assign_regions_to_voronoi_cells(poly_objs, voronoi_cells)

    zones: List[List[int]] = []
    grouped: Dict[int, List[int]] = {}
    for rid, zid in enumerate(zone_id):
        if zid < 0:
            continue
        grouped.setdefault(zid, []).append(rid)
    for zid in sorted(grouped.keys()):
        members = grouped[zid]
        if not members:
            continue
        member_set = set(members)
        seen: set[int] = set()
        components: List[List[int]] = []
        for rid in members:
            if rid in seen:
                continue
            stack = [rid]
            comp: List[int] = []
            seen.add(rid)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in adj[cur]:
                    if nb in member_set and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            components.append(comp)
        if not components:
            continue
        components.sort(
            key=lambda comp: sum(poly_objs[r].area for r in comp if not poly_objs[r].is_empty),
            reverse=True,
        )
        for comp in components:
            zones.append(comp)

    zone_id = [-1] * len(polys)
    for zid, members in enumerate(zones):
        for rid in members:
            zone_id[rid] = zid

    remaining = [rid for rid, zid in enumerate(zone_id) if zid == -1 and not poly_objs[rid].is_empty]
    for rid in remaining:
        c = poly_objs[rid].centroid
        best = None
        best_d = 1e18
        for zid, members in enumerate(zones):
            if not members:
                continue
            if not any(nb in members for nb in adj[rid]):
                continue
            cm = unary_union([poly_objs[m] for m in members]).centroid
            d = (cm.x - c.x) ** 2 + (cm.y - c.y) ** 2
            if d < best_d:
                best_d = d
                best = zid
        if best is None:
            zones.append([rid])
            zone_id[rid] = len(zones) - 1
        else:
            zones[best].append(rid)
            zone_id[rid] = best

    while len(zones) > target and zones:
        smallest = min(range(len(zones)), key=lambda i: sum(poly_objs[r].area for r in zones[i]))
        if len(zones) == 1:
            break
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
        for i, members in enumerate(zones):
            for rid in members:
                zone_id[rid] = i

    return zone_id


def _remap_zones_by_area(
    polys: List[List[Tuple[float, float]]],
    zone_id: List[int],
) -> Tuple[List[int], Dict[int, List[int]]]:
    if not zone_id:
        return zone_id, {}
    zones: Dict[int, List[int]] = {}
    for rid, zid in enumerate(zone_id):
        if zid < 0:
            continue
        zones.setdefault(zid, []).append(rid)
    if not zones:
        return zone_id, {}
    areas: Dict[int, float] = {}
    for zid, members in zones.items():
        areas[zid] = sum(Polygon(polys[r]).area for r in members)
    order = sorted(zones.keys(), key=lambda z: areas.get(z, 0.0), reverse=True)
    remap = {old: new for new, old in enumerate(order)}
    new_zone_id = [remap.get(zid, -1) if zid >= 0 else -1 for zid in zone_id]
    region_area = [Polygon(polys[r]).area for r in range(len(polys))]
    new_zones: Dict[int, List[int]] = {}
    for rid, zid in enumerate(new_zone_id):
        if zid < 0:
            continue
        new_zones.setdefault(zid, []).append(rid)
    for zid, members in new_zones.items():
        members.sort(key=lambda r: region_area[r], reverse=True)
    return new_zone_id, new_zones


def save_zones_cache(zone_id: List[int], polys: List[List[Tuple[float, float]]], out_path) -> None:
    zones: Dict[int, List[int]] = {}
    for rid, zid in enumerate(zone_id):
        zones.setdefault(zid, []).append(rid)
    data = {
        "zones": {str(k): v for k, v in zones.items()},
        "polys": polys,
        "svg_mtime": config.SVG_PATH.stat().st_mtime,
    }
    out_path.write_text(json.dumps(data), encoding="utf-8")


def load_zones_cache(path) -> Tuple[List[List[Tuple[float, float]]], List[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    polys = [list(map(lambda p: (p[0], p[1]), pts)) for pts in data["polys"]]
    zone_id = [-1] * len(polys)
    for k, members in data["zones"].items():
        zid = int(k)
        for rid in members:
            zone_id[rid] = zid
    return polys, zone_id


def write_zones_log(zone_id: List[int], out_path, zone_labels: Dict[int, int] | None = None) -> None:
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
            zone_polys.append([(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)])
            debug["empty"].append(zid)
            continue
        if merged.geom_type != "Polygon":
            merged = merged.convex_hull
            debug["convex_hull"].append(zid)
        zone_polys.append(list(merged.exterior.coords))
    return zone_polys, zone_order, debug


def build_zone_geoms(polys: List[List[Tuple[float, float]]], zone_id: List[int]) -> Dict[int, BaseGeometry]:
    zone_geoms: Dict[int, BaseGeometry] = {}
    zones: Dict[int, List[Polygon]] = {}
    for rid, zid in enumerate(zone_id):
        zones.setdefault(zid, []).append(Polygon(polys[rid]))
    for zid, parts in zones.items():
        zone_geoms[zid] = unary_union(parts)
    return zone_geoms


def _label_pos_for_zone(geom: BaseGeometry) -> Tuple[float, float]:
    if geom.is_empty:
        return (0.0, 0.0)
    c = geom.centroid
    return float(c.x), float(c.y)


def _label_pos_outside(geom: BaseGeometry, offset: float) -> Tuple[float, float]:
    if geom.is_empty:
        return (0.0, 0.0)
    minx, miny, maxx, maxy = geom.bounds
    return maxx + offset, (miny + maxy) / 2.0


def _snap_key(pt: Tuple[float, float], snap: float) -> Tuple[int, int]:
    if snap <= 0:
        return (int(round(pt[0])), int(round(pt[1])))
    return (int(round(pt[0] / snap)), int(round(pt[1] / snap)))


def build_zone_boundaries(
    polys: List[List[Tuple[float, float]]],
    zone_id: List[int],
    snap: float = config.EDGE_EPS,
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
            used[i] = True
            path_keys = [k1, k2]

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
    zone_scale = config.DRAW_SCALE * 2
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
            config.LABEL_FONT_SCALE,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    img_out = cv2.resize(img, (int(w * config.DRAW_SCALE), int(h * config.DRAW_SCALE)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(config.OUT_ZONE_PNG), img_out)
