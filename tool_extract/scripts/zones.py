from __future__ import annotations

import json
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from . import config


def build_zones(polys: List[List[Tuple[float, float]]], target: int) -> List[int]:
    if not polys:
        return []
    minx = min(p[0] for poly in polys for p in poly)
    miny = min(p[1] for poly in polys for p in poly)
    maxx = max(p[0] for poly in polys for p in poly)
    maxy = max(p[1] for poly in polys for p in poly)
    cell_w = (maxx - minx) / config.GRID_X
    cell_h = (maxy - miny) / config.GRID_Y

    zones: List[List[int]] = []
    zone_id = [-1] * len(polys)
    poly_objs = [Polygon(p) for p in polys]

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

    for gy in range(config.GRID_Y):
        for gx in range(config.GRID_X):
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

    split_zones: List[List[int]] = []
    for members in zones:
        if not members:
            continue
        member_set = set(members)
        seen: set[int] = set()
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
            split_zones.append(comp)
    zones = split_zones
    zone_id = [-1] * len(polys)
    for zid, members in enumerate(zones):
        for rid in members:
            zone_id[rid] = zid

    remaining = [rid for rid, zid in enumerate(zone_id) if zid == -1]
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
