from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from math import ceil
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np

PACK_DEBUG_STATS: Dict[str, int] | None = None
from shapely.affinity import rotate as _srotate, translate as _stranslate
from shapely.geometry import Polygon, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from PIL import Image, ImageDraw

import config
import geometry
import svg_utils
import zones

try:
    from shapely.validation import make_valid
except Exception:  # pragma: no cover
    def make_valid(geom):
        return geom.buffer(0)


def _log_step(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    # Avoid failing API requests when stdout handle is transiently invalid
    # (seen intermittently under Flask dev server/reloader on Windows).
    try:
        print(f"[{ts}] {msg}")
    except OSError:
        pass


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


def _offset_outline_same_vertices(
    pts: List[Tuple[float, float]], offset: float
) -> List[Tuple[float, float]]:
    if len(pts) < 3:
        return pts[:]
    # Determine outward normals using polygon orientation.
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += (x1 * y2) - (x2 * y1)
    ccw = area > 0

    # Build offset lines for each edge (p0->p1).
    lines = []
    for i in range(len(pts)):
        p0 = pts[i]
        p1 = pts[(i + 1) % len(pts)]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        ln = float(np.hypot(dx, dy))
        if ln <= 1e-6:
            lines.append(None)
            continue
        # Outward normal (right-hand for CCW, left-hand for CW)
        if ccw:
            nx, ny = dy / ln, -dx / ln
        else:
            nx, ny = -dy / ln, dx / ln
        q0 = (p0[0] + nx * offset, p0[1] + ny * offset)
        q1 = (p1[0] + nx * offset, p1[1] + ny * offset)
        lines.append((q0, q1, (nx, ny)))

    # Intersect consecutive offset lines to get new vertices.
    out = []
    for i in range(len(pts)):
        prev = lines[(i - 1) % len(pts)]
        cur = lines[i]
        if prev is None or cur is None:
            out.append(pts[i])
            continue
        (p1, p2, n1) = prev
        (p3, p4, n2) = cur
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-8:
            # Nearly parallel: shift original vertex by current normal.
            out.append((pts[i][0] + n2[0] * offset, pts[i][1] + n2[1] * offset))
            continue
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        out.append((px, py))
    return out


def _point_in_poly(pt: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = (x2 - x1) * (y - y1) / (y2 - y1 + 0.0) + x1
            if x < xint:
                inside = not inside
    return inside


def _build_zone_pack_polys(
    zone_polys: List[List[Tuple[float, float]]],
    bleed: float,
    bevel_angle: float = 60.0,
) -> List[List[Tuple[float, float]]]:
    if not zone_polys:
        return []
    out: List[List[Tuple[float, float]]] = []
    for poly in zone_polys:
        pts = poly[:]
        if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) < 1e-6 and abs(pts[0][1] - pts[-1][1]) < 1e-6:
            pts = pts[:-1]
        if bleed > 0 and len(pts) >= 3:
            pts = _offset_outline_same_vertices(pts, bleed)
            pts, _ = _bevel_outline_by_angle(pts, bleed, angle_thresh=bevel_angle)
        out.append(pts)
    return out


def _min_bbox_align_angle(poly: List[Tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    pg = Polygon(poly)
    if pg.is_empty:
        return 0.0
    rect = pg.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return 0.0
    best_len = -1.0
    best_ang = 0.0
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        ln = float(np.hypot(dx, dy))
        if ln > best_len:
            best_len = ln
            best_ang = float(np.degrees(np.arctan2(dy, dx)))
    # Rotate so the longest edge aligns to +X.
    return -best_ang


def _resolve_pack_overlaps(
    zone_polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[float, float, int, int, bool]],
    rot_info: List[Dict[str, float]],
    step: float,
    padding: float = 0.0,
    max_iter: int = 50,
) -> List[Tuple[float, float, int, int, bool]]:
    if not zone_polys or not placements:
        return placements
    n = min(len(zone_polys), len(placements))
    step = max(0.5, float(step))
    pad = max(0.0, float(padding))
    out = list(placements)
    for _ in range(max_iter):
        moved = False
        tpolys: List[Polygon] = []
        centroids: List[Tuple[float, float]] = []
        for i in range(n):
            dx, dy, w, h, rflag = out[i]
            info = rot_info[i] if i < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
            ang = float(info.get("angle", 0.0))
            cx = float(info.get("cx", 0.0))
            cy = float(info.get("cy", 0.0))
            rpts = _rotate_pts(zone_polys[i], ang, cx, cy)
            tpts = [(p[0] + dx, p[1] + dy) for p in rpts]
            poly = Polygon(tpts)
            if pad > 0:
                poly = poly.buffer(pad)
            tpolys.append(poly)
            if poly.is_empty:
                centroids.append((0.0, 0.0))
            else:
                c = poly.centroid
                centroids.append((float(c.x), float(c.y)))
        for i in range(n):
            pi = tpolys[i]
            if pi.is_empty:
                continue
            for j in range(i + 1, n):
                pj = tpolys[j]
                if pj.is_empty:
                    continue
                b1 = pi.bounds
                b2 = pj.bounds
                if b2[0] > b1[2] or b2[2] < b1[0] or b2[1] > b1[3] or b2[3] < b1[1]:
                    continue
                if not pi.intersects(pj):
                    continue
                moved = True
                vx = centroids[j][0] - centroids[i][0]
                vy = centroids[j][1] - centroids[i][1]
                ln = float(np.hypot(vx, vy))
                if ln <= 1e-6:
                    vx, vy = 1.0, 0.0
                    ln = 1.0
                ux, uy = vx / ln, vy / ln
                # Adaptive push distance: keep conservative to avoid sparse layout.
                push = step
                try:
                    inter = pi.intersection(pj)
                    if not inter.is_empty:
                        ia = float(getattr(inter, "area", 0.0) or 0.0)
                        if ia > 0:
                            push += min(2.0, np.sqrt(ia) * 0.15)
                except Exception:
                    pass
                dxj, dyj, wj, hj, rfj = out[j]
                # Move only one item to reduce global drift/fragmentation.
                out[j] = (dxj + ux * push, dyj + uy * push, wj, hj, rfj)
        if not moved:
            break
    return out


def _fit_placements_into_canvas(
    placements: List[Tuple[float, float, int, int, bool]],
    rot_info: List[Dict[str, float]],
    canvas: Tuple[int, int],
) -> List[Tuple[float, float, int, int, bool]]:
    """Translate placed items per bin so they stay inside canvas bounds."""
    if not placements or not rot_info:
        return placements
    w, h = float(canvas[0]), float(canvas[1])
    out = list(placements)
    n = min(len(out), len(rot_info))
    bins: Dict[int, List[int]] = {}
    for i in range(n):
        dx, dy, bw, bh, _ = out[i]
        if bw <= 0 or bh <= 0:
            continue
        b = int(rot_info[i].get("bin", 0))
        bins.setdefault(b, []).append(i)
    for _, idxs in bins.items():
        minx = None
        miny = None
        maxx = None
        maxy = None
        for i in idxs:
            dx, dy, bw, bh, _ = out[i]
            info = rot_info[i]
            x0 = float(info.get("minx", 0.0)) + float(dx)
            y0 = float(info.get("miny", 0.0)) + float(dy)
            x1 = x0 + float(bw)
            y1 = y0 + float(bh)
            minx = x0 if minx is None else min(minx, x0)
            miny = y0 if miny is None else min(miny, y0)
            maxx = x1 if maxx is None else max(maxx, x1)
            maxy = y1 if maxy is None else max(maxy, y1)
        if minx is None or miny is None or maxx is None or maxy is None:
            continue
        tx = 0.0
        ty = 0.0
        if minx < 0.0:
            tx += -minx
        if (maxx + tx) > w:
            tx += (w - (maxx + tx))
        if miny < 0.0:
            ty += -miny
        if (maxy + ty) > h:
            ty += (h - (maxy + ty))
        if abs(tx) > 1e-6 or abs(ty) > 1e-6:
            for i in idxs:
                dx, dy, bw, bh, rf = out[i]
                out[i] = (dx + tx, dy + ty, bw, bh, rf)
    return out


def _bevel_poly(points: List[Tuple[float, float]], r: float) -> List[Tuple[float, float]]:
    if r <= 0 or len(points) < 3:
        return points
    out_pts: List[Tuple[float, float]] = []
    n = len(points)
    for i in range(n):
        p_prev = points[(i - 1) % n]
        p_cur = points[i]
        p_next = points[(i + 1) % n]
        v1x = p_prev[0] - p_cur[0]
        v1y = p_prev[1] - p_cur[1]
        v2x = p_next[0] - p_cur[0]
        v2y = p_next[1] - p_cur[1]
        l1 = float(np.hypot(v1x, v1y))
        l2 = float(np.hypot(v2x, v2y))
        if l1 <= 1e-6 or l2 <= 1e-6:
            out_pts.append(p_cur)
            continue
        u1x, u1y = v1x / l1, v1y / l1
        u2x, u2y = v2x / l2, v2y / l2
        p1 = (p_cur[0] + u1x * r, p_cur[1] + u1y * r)
        p2 = (p_cur[0] + u2x * r, p_cur[1] + u2y * r)
        out_pts.append(p1)
        out_pts.append(p2)
    return out_pts


def _bevel_poly_sharp(
    points: List[Tuple[float, float]], r: float, angle_thresh: float
) -> List[Tuple[float, float]]:
    if r <= 0 or len(points) < 3:
        return points
    out_pts: List[Tuple[float, float]] = []
    n = len(points)
    for i in range(n):
        p_prev = points[(i - 1) % n]
        p_cur = points[i]
        p_next = points[(i + 1) % n]
        v1x = p_prev[0] - p_cur[0]
        v1y = p_prev[1] - p_cur[1]
        v2x = p_next[0] - p_cur[0]
        v2y = p_next[1] - p_cur[1]
        l1 = float(np.hypot(v1x, v1y))
        l2 = float(np.hypot(v2x, v2y))
        if l1 <= 1e-6 or l2 <= 1e-6:
            out_pts.append(p_cur)
            continue
        u1x, u1y = v1x / l1, v1y / l1
        u2x, u2y = v2x / l2, v2y / l2
        dot = max(-1.0, min(1.0, u1x * u2x + u1y * u2y))
        angle = float(np.degrees(np.arccos(dot)))
        # Determine interior angle (concave needs 360 - angle)
        cross = u1x * u2y - u1y * u2x
        interior = 360.0 - angle if cross < 0 else angle
        if interior < angle_thresh:
            p1 = (p_cur[0] + u1x * r, p_cur[1] + u1y * r)
            p2 = (p_cur[0] + u2x * r, p_cur[1] + u2y * r)
            out_pts.append(p1)
            out_pts.append(p2)
        else:
            out_pts.append(p_cur)
    return out_pts


def _bevel_corner(
    prev_pt: Tuple[float, float],
    cur_pt: Tuple[float, float],
    next_pt: Tuple[float, float],
    r: float,
) -> List[Tuple[float, float]]:
    if r <= 0:
        return [cur_pt]
    v1x = prev_pt[0] - cur_pt[0]
    v1y = prev_pt[1] - cur_pt[1]
    v2x = next_pt[0] - cur_pt[0]
    v2y = next_pt[1] - cur_pt[1]
    l1 = float(np.hypot(v1x, v1y))
    l2 = float(np.hypot(v2x, v2y))
    if l1 <= 1e-6 or l2 <= 1e-6:
        return [cur_pt]
    d1 = min(r, l1 * 0.49)
    d2 = min(r, l2 * 0.49)
    p1 = (cur_pt[0] + v1x / l1 * d1, cur_pt[1] + v1y / l1 * d1)
    p2 = (cur_pt[0] + v2x / l2 * d2, cur_pt[1] + v2y / l2 * d2)
    return [p1, p2]


def _snap_key_local(pt: Tuple[float, float], snap: float) -> Tuple[int, int]:
    if snap <= 0:
        return (int(round(pt[0])), int(round(pt[1])))
    return (int(round(pt[0] / snap)), int(round(pt[1] / snap)))


def _build_boundary_from_polys(
    polys: List[List[Tuple[float, float]]], snap: float
) -> List[List[Tuple[float, float]]]:
    edge_counts: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
    point_sum: Dict[Tuple[int, int], List[float]] = {}

    def _acc_point(k: Tuple[int, int], p: Tuple[float, float]) -> None:
        if k not in point_sum:
            point_sum[k] = [0.0, 0.0, 0.0]
        point_sum[k][0] += p[0]
        point_sum[k][1] += p[1]
        point_sum[k][2] += 1.0

    for pts in polys:
        if len(pts) < 2:
            continue
        n = len(pts)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            k1 = _snap_key_local(p1, snap)
            k2 = _snap_key_local(p2, snap)
            if k1 == k2:
                continue
            _acc_point(k1, p1)
            _acc_point(k2, p2)
            ek = (k1, k2) if k1 < k2 else (k2, k1)
            edge_counts[ek] = edge_counts.get(ek, 0) + 1

    boundary_edges = [ek for ek, c in edge_counts.items() if c == 1]
    if not boundary_edges:
        return []

    point_map: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for k, (sx, sy, cnt) in point_sum.items():
        if cnt > 0:
            point_map[k] = (sx / cnt, sy / cnt)

    adj: Dict[Tuple[int, int], List[int]] = {}
    for idx, (k1, k2) in enumerate(boundary_edges):
        adj.setdefault(k1, []).append(idx)
        adj.setdefault(k2, []).append(idx)

    used = [False] * len(boundary_edges)
    polylines: List[List[Tuple[float, float]]] = []

    def _next_edge(cur_key: Tuple[int, int], prev_key: Tuple[int, int] | None) -> int | None:
        candidates = [ei for ei in adj.get(cur_key, []) if not used[ei]]
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
            a, b = boundary_edges[ei]
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

    for i, (k1, k2) in enumerate(boundary_edges):
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
            a, b = boundary_edges[next_edge]
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
            a, b = boundary_edges[next_edge]
            nxt = b if a == cur else a
            if nxt == path_keys[0]:
                break
            path_keys.insert(0, nxt)
            if nxt == path_keys[-1]:
                break

        path_pts = [point_map[k] for k in path_keys if k in point_map]
        if len(path_pts) >= 2:
            polylines.append(path_pts)

    return polylines


def _poly_area_abs(pts: List[Tuple[float, float]]) -> float:
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def _max_edge_len(pts: List[Tuple[float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    max_len = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        d = float(np.hypot(x2 - x1, y2 - y1))
        if d > max_len:
            max_len = d
    return max_len


def _edge_counts_from_outlines(
    outlines: List[List[Tuple[float, float]]], snap: float
) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], int]:
    edge_counts: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
    for polyline in outlines:
        if len(polyline) < 2:
            continue
        closed = abs(polyline[0][0] - polyline[-1][0]) < 1e-6 and abs(polyline[0][1] - polyline[-1][1]) < 1e-6
        pts = polyline[:-1] if closed and len(polyline) > 1 else polyline
        n = len(pts)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n] if closed else pts[i + 1] if i + 1 < n else None
            if p2 is None:
                continue
            k1 = _snap_key_local(p1, snap)
            k2 = _snap_key_local(p2, snap)
            if k1 == k2:
                continue
            ek = (k1, k2) if k1 < k2 else (k2, k1)
            edge_counts[ek] = edge_counts.get(ek, 0) + 1
    return edge_counts


def _free_keys_for_outline(
    polyline: List[Tuple[float, float]],
    edge_counts: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int],
    snap: float,
) -> set[Tuple[int, int]]:
    free_keys: set[Tuple[int, int]] = set()
    if len(polyline) < 3:
        return free_keys
    closed = abs(polyline[0][0] - polyline[-1][0]) < 1e-6 and abs(polyline[0][1] - polyline[-1][1]) < 1e-6
    pts = polyline[:-1] if closed and len(polyline) > 1 else polyline
    n = len(pts)
    if n < 3:
        return free_keys
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_cur = pts[i]
        p_next = pts[(i + 1) % n]
        k_prev = _snap_key_local(p_prev, snap)
        k_cur = _snap_key_local(p_cur, snap)
        k_next = _snap_key_local(p_next, snap)
        e1 = (k_prev, k_cur) if k_prev < k_cur else (k_cur, k_prev)
        e2 = (k_cur, k_next) if k_cur < k_next else (k_next, k_cur)
        if edge_counts.get(e1, 0) == 1 and edge_counts.get(e2, 0) == 1:
            free_keys.add(k_cur)
    return free_keys


def _bevel_outline_by_angle(
    polyline: List[Tuple[float, float]],
    r: float,
    angle_thresh: float = 90.0,
    free_keys: set[Tuple[int, int]] | None = None,
    snap: float = 0.0,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    if r <= 0 or len(polyline) < 3:
        return polyline, []
    closed = abs(polyline[0][0] - polyline[-1][0]) < 1e-6 and abs(polyline[0][1] - polyline[-1][1]) < 1e-6
    pts = polyline[:-1] if closed and len(polyline) > 1 else polyline
    n = len(pts)
    if n < 3:
        return polyline, []
    out: List[Tuple[float, float]] = []
    debug_pts: List[Tuple[float, float]] = []
    for i in range(n):
        prev_pt = pts[(i - 1) % n]
        cur_pt = pts[i]
        next_pt = pts[(i + 1) % n]
        key = _snap_key_local(cur_pt, snap) if free_keys is not None else None
        if free_keys is not None and key not in free_keys:
            out.append(cur_pt)
            continue
        v1x = prev_pt[0] - cur_pt[0]
        v1y = prev_pt[1] - cur_pt[1]
        v2x = next_pt[0] - cur_pt[0]
        v2y = next_pt[1] - cur_pt[1]
        l1 = float(np.hypot(v1x, v1y))
        l2 = float(np.hypot(v2x, v2y))
        if l1 > 1e-6 and l2 > 1e-6:
            u1x, u1y = v1x / l1, v1y / l1
            u2x, u2y = v2x / l2, v2y / l2
            dot = max(-1.0, min(1.0, u1x * u2x + u1y * u2y))
            angle = float(np.degrees(np.arccos(dot)))
            if angle < angle_thresh:
                # Smaller angle => much larger bevel (non-linear).
                angle_safe = max(angle, 1e-3)
                scale = (angle_thresh / angle_safe) ** 1.5
                r_eff = min(r * scale, r * 6.0)
                out.extend(_bevel_corner(prev_pt, cur_pt, next_pt, r_eff))
                debug_pts.append(cur_pt)
                continue
        out.append(cur_pt)
    if closed:
        out.append(out[0])
    return out, debug_pts


def _rotate_zone_transforms_180(
    zone_shift: Dict[int, Tuple[float, float]],
    zone_center: Dict[int, Tuple[float, float]],
    zone_rot: Dict[int, float],
    canvas: Tuple[int, int],
) -> None:
    w, h = canvas
    cx_canvas = w / 2.0
    cy_canvas = h / 2.0
    for zid, (dx, dy) in list(zone_shift.items()):
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        ndx = (2 * cx_canvas) - (2 * cx) - float(dx)
        ndy = (2 * cy_canvas) - (2 * cy) - float(dy)
        zone_shift[zid] = (ndx, ndy)
        zone_rot[zid] = float(zone_rot.get(zid, 0.0)) + 180.0


def pack_regions(
    polys: List[List[Tuple[float, float]]],
    canvas: Tuple[int, int],
    allow_rotate: bool = True,
    angle_step: float = 5.0,
    grid_step: float = 5.0,
    fixed_angles: List[float] | None = None,
    fixed_centers: List[Tuple[float, float]] | None = None,
    max_bins: int = 2,
    try_heuristics: bool = False,
    two_pass: bool = False,
    preferred_indices: List[int] | None = None,
    use_gap_only: bool = False,
) -> Tuple[List[Tuple[int, int, int, int, bool]], List[int], List[Dict[str, float]]]:
    global PACK_DEBUG_STATS
    # Rectpack has been fully retired. Keep this API but route to raster packer.
    PACK_DEBUG_STATS = {
        "rectpack_placed": 0,
        "gap_right_added": 0,
        "gap_bottom_added": 0,
        "total_placed": 0,
        "remaining_after_gap": 0,
    }
    rotations = [0.0, 90.0, 180.0, 270.0] if allow_rotate else [0.0]
    placements, order, rot_info = pack_regions_raster_fast(
        polys,
        canvas,
        fixed_centers=fixed_centers,
        grid_step=max(2.0, float(grid_step)),
        rotations=rotations,
        search_stride=1,
        safety_padding=max(
            1.0,
            float(getattr(config, "PADDING", 0.0)),
            float(getattr(config, "PACK_BLEED", 0.0)) * 0.5,
        ),
    )
    PACK_DEBUG_STATS["total_placed"] = len(order)
    PACK_DEBUG_STATS["remaining_after_gap"] = max(0, len(polys) - len(order))
    return placements, order, rot_info

    w, h = canvas
    pad = float(config.PADDING)
    x_min = config.PACK_MARGIN_X
    y_min = config.PACK_MARGIN_Y
    x_max = w - config.PACK_MARGIN_X
    y_max = h - config.PACK_MARGIN_Y

    bboxes: List[Tuple[int, float, float, int, int]] = []
    rot_info: List[Dict[str, float]] = []
    bounds0: List[Tuple[float, float, int, int]] = []
    bounds90: List[Tuple[float, float, int, int]] = []
    for i, pts in enumerate(polys):
        poly = Polygon(pts)
        if poly.is_empty:
            x0 = y0 = 0.0
            x1 = y1 = 1.0
            angle = 0.0
            cx = cy = 0.0
        else:
            if fixed_centers is not None and i < len(fixed_centers):
                cx, cy = fixed_centers[i]
            else:
                cx, cy = float(poly.centroid.x), float(poly.centroid.y)
            angle = 0.0
            best_area = 1e18
            best_bounds = None
            if fixed_angles is not None and i < len(fixed_angles):
                ang = float(fixed_angles[i])
                rpts = _rotate_pts(pts, ang, cx, cy)
                xs = [p[0] for p in rpts]
                ys = [p[1] for p in rpts]
                x0t, y0t, x1t, y1t = min(xs), min(ys), max(xs), max(ys)
                best_bounds = (x0t, y0t, x1t, y1t)
                angle = ang
            elif allow_rotate:
                ang = 0.0
                while ang <= 180.0:
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
        rot_info.append({"angle": float(angle), "cx": float(cx), "cy": float(cy), "minx": x0, "miny": y0})
        # Precompute 0/90 bounds for gap-only packing.
        if poly.is_empty or not pts:
            bounds0.append((0.0, 0.0, 1, 1))
            bounds90.append((0.0, 0.0, 1, 1))
        else:
            r0 = _rotate_pts(pts, 0.0, cx, cy)
            xs0 = [p[0] for p in r0]
            ys0 = [p[1] for p in r0]
            x0_0, y0_0, x1_0, y1_0 = min(xs0), min(ys0), max(xs0), max(ys0)
            bw0 = int(ceil((x1_0 - x0_0) + pad * 2))
            bh0 = int(ceil((y1_0 - y0_0) + pad * 2))
            bounds0.append((x0_0, y0_0, bw0, bh0))
            r90 = _rotate_pts(pts, 90.0, cx, cy)
            xs90 = [p[0] for p in r90]
            ys90 = [p[1] for p in r90]
            x0_90, y0_90, x1_90, y1_90 = min(xs90), min(ys90), max(xs90), max(ys90)
            bw90 = int(ceil((x1_90 - x0_90) + pad * 2))
            bh90 = int(ceil((y1_90 - y0_90) + pad * 2))
            bounds90.append((x0_90, y0_90, bw90, bh90))

    if use_gap_only:
        placements: List[Tuple[int, int, int, int, bool]] = [(-1, -1, 0, 0, False)] * len(polys)
        order: List[int] = []
        # Sort by area desc (use larger of 0/90).
        indices = list(range(len(bboxes)))
        indices.sort(
            key=lambda i: max(bounds0[i][2] * bounds0[i][3], bounds90[i][2] * bounds90[i][3]),
            reverse=True,
        )
        placed_ids: List[int] = []
        right_y = 0.0
        bottom_x = 0.0
        gap_right_added = 0
        gap_bottom_added = 0

        def _recompute_used() -> Tuple[float, float]:
            umax_x = 0.0
            umax_y = 0.0
            for pid in placed_ids:
                dx, dy, bw, bh, _rot = placements[pid]
                info = rot_info[pid]
                x = dx + float(info.get("minx", 0.0))
                y = dy + float(info.get("miny", 0.0))
                umax_x = max(umax_x, x + bw)
                umax_y = max(umax_y, y + bh)
            return umax_x, umax_y

        for rid in indices:
            used_max_x, used_max_y = _recompute_used()
            right_x = used_max_x
            bottom_y = used_max_y
            right_w = max(0.0, w - right_x)
            bottom_w = max(0.0, w - right_w)
            best = None
            for ang, (x0, y0, bw, bh) in [(0.0, bounds0[rid]), (90.0, bounds90[rid])]:
                if bw <= 0 or bh <= 0:
                    continue
                # right gap placement
                if bw <= right_w and right_y + bh <= h:
                    new_used_x = max(used_max_x, right_x + bw)
                    new_used_y = max(used_max_y, right_y + bh)
                    metric = (w - new_used_x) + (h - new_used_y)
                    best = (metric, "right", ang, x0, y0, bw, bh, right_x, right_y) if best is None or metric < best[0] else best
                # bottom gap placement
                if bw <= bottom_w and bottom_x + bw <= bottom_w and bh <= (h - bottom_y):
                    new_used_x = max(used_max_x, bottom_x + bw)
                    new_used_y = max(used_max_y, bottom_y + bh)
                    metric = (w - new_used_x) + (h - new_used_y)
                    best = (metric, "bottom", ang, x0, y0, bw, bh, bottom_x, bottom_y) if best is None or metric < best[0] else best
            if best is None:
                continue
            _metric, loc, ang, x0, y0, bw, bh, px, py = best
            placements[rid] = (int(px - x0), int(py - y0), int(bw), int(bh), False)
            rot_info[rid]["angle"] = float(ang)
            rot_info[rid]["minx"] = float(x0)
            rot_info[rid]["miny"] = float(y0)
            rot_info[rid]["bin"] = 0
            placed_ids.append(rid)
            order.append(rid)
            if loc == "right":
                right_y += bh
                gap_right_added += 1
            else:
                bottom_x += bw
                gap_bottom_added += 1

        PACK_DEBUG_STATS = {
            "rectpack_placed": 0,
            "gap_right_added": gap_right_added,
            "gap_bottom_added": gap_bottom_added,
            "total_placed": len(placed_ids),
            "remaining_after_gap": len(indices) - len(placed_ids),
        }
        return placements, order, rot_info

    bin_w = int(x_max - x_min)
    bin_h = int(y_max - y_min)

    def _order_indices(indices: List[int]) -> List[int]:
        if not preferred_indices:
            return indices
        pref = [i for i in preferred_indices if i in indices]
        rest = [i for i in indices if i not in set(pref)]
        return pref + rest

    def _run_packer(pack_algo=None, sort_algo=None, rect_indices: List[int] | None = None):
        if pack_algo is None and sort_algo is None:
            packer = newPacker(rotation=False)
        else:
            packer = newPacker(rotation=False, pack_algo=pack_algo, sort_algo=sort_algo)
        packer.add_bin(bin_w, bin_h)
        indices = rect_indices if rect_indices is not None else list(range(len(bboxes)))
        indices = _order_indices(indices)
        for idx in indices:
            _, _, _, bw, bh = bboxes[idx]
            packer.add_rect(bw, bh, rid=idx)
        packer.pack()
        return list(packer.rect_list())

    def _layout_score(rects: List[Tuple[int, int, int, int, int, int]]) -> Tuple[float, ...]:
        """Higher is better: placed-count, density, compactness, aspect stability."""
        placed = len(rects)
        if placed <= 0:
            return (0.0, -1e18, -1e18, -1e18)
        xs0 = [float(x) for _, x, _, _, _, _ in rects]
        ys0 = [float(y) for _, _, y, _, _, _ in rects]
        xs1 = [float(x + pw) for _, x, _, pw, _, _ in rects]
        ys1 = [float(y + ph) for _, _, y, _, ph, _ in rects]
        minx = min(xs0)
        miny = min(ys0)
        maxx = max(xs1)
        maxy = max(ys1)
        used_w = max(1.0, maxx - minx)
        used_h = max(1.0, maxy - miny)
        used_area = used_w * used_h
        placed_area = float(sum(max(0, pw) * max(0, ph) for _, _, _, pw, ph, _ in rects))
        density = placed_area / used_area if used_area > 0 else 0.0
        # Prefer used bbox close to bin aspect ratio to avoid long "L / strip-like" distributions.
        bin_aspect = float(bin_w) / float(max(1, bin_h))
        used_aspect = used_w / used_h
        aspect_penalty = abs(used_aspect - bin_aspect)
        # Lexicographic score: place all first, then denser, then smaller area, then aspect.
        return (float(placed), density, -used_area, -aspect_penalty)

    def _apply_rects(
        rects: List[Tuple[int, int, int, int, int, int]],
        bin_index_offset: int,
        base_bin_map: Dict[int, int] | None = None,
    ) -> List[int]:
        bin_bounds: Dict[int, Tuple[int, int, int, int]] = {}
        for bin_idx, x, y, pw, ph, _ in rects:
            if bin_idx not in bin_bounds:
                bin_bounds[bin_idx] = (x, y, x + pw, y + ph)
            else:
                x0, y0, x1, y1 = bin_bounds[bin_idx]
                bin_bounds[bin_idx] = (
                    min(x0, x),
                    min(y0, y),
                    max(x1, x + pw),
                    max(y1, y + ph),
                )
        bin_offsets: Dict[int, Tuple[int, int]] = {}
        for bin_idx, (bx0, by0, bx1, by1) in bin_bounds.items():
            off_x = 0
            off_y = 0
            bin_offsets[bin_idx] = (off_x - bx0, off_y - by0)
        placed_ids: List[int] = []
        for bin_idx, x, y, pw, ph, rid in rects:
            orig = bboxes[rid]
            x0 = orig[1]
            y0 = orig[2]
            off_x, off_y = bin_offsets.get(bin_idx, (int(x_min), int(y_min)))
            placements[rid] = (
                int(x + off_x + pad - int(x0)),
                int(y + off_y + pad - int(y0)),
                int(pw),
                int(ph),
                False,
            )
            rot_info[rid]["bin"] = int(bin_idx + bin_index_offset)
            if base_bin_map is not None:
                rot_info[rid]["base_bin"] = int(base_bin_map.get(rid, bin_idx))
            placed_ids.append(rid)
        return placed_ids

    if max_bins == 1 and try_heuristics and not preferred_indices:
        combos = [
            (None, None),
            (maxrects.MaxRectsBssf, rectpack.SORT_AREA),
            (maxrects.MaxRectsBlsf, rectpack.SORT_AREA),
            (maxrects.MaxRectsBaf, rectpack.SORT_AREA),
            (maxrects.MaxRectsBl, rectpack.SORT_AREA),
            (maxrects.MaxRectsBssf, rectpack.SORT_LSIDE),
            (maxrects.MaxRectsBlsf, rectpack.SORT_LSIDE),
            (maxrects.MaxRectsBaf, rectpack.SORT_LSIDE),
            (maxrects.MaxRectsBssf, rectpack.SORT_SSIDE),
            (maxrects.MaxRectsBlsf, rectpack.SORT_SSIDE),
            (maxrects.MaxRectsBaf, rectpack.SORT_SSIDE),
        ]
        best: List[Tuple[int, int, int, int, int, int]] = []
        best_score: Tuple[float, ...] | None = None
        for pack_algo, sort_algo in combos:
            rects_try = _run_packer(pack_algo, sort_algo)
            score_try = _layout_score(rects_try)
            if best_score is None or score_try > best_score:
                best = rects_try
                best_score = score_try
        rects = best
    else:
        packer = newPacker(rotation=False)
        for _ in range(max_bins):
            packer.add_bin(bin_w, bin_h)
        indices = _order_indices(list(range(len(bboxes))))
        for idx in indices:
            _, _, _, bw, bh = bboxes[idx]
            packer.add_rect(bw, bh, rid=idx)
        packer.pack()
        rects = list(packer.rect_list())

    placements: List[Tuple[int, int, int, int, bool]] = [(-1, -1, 0, 0, False)] * len(polys)
    order: List[int] = []
    if not rects:
        return placements, order, rot_info

    rectpack_placed = 0
    gap_right_added = 0
    gap_bottom_added = 0
    base_bin_map: Dict[int, int] | None = None
    if two_pass:
        # If we already have multi-bin rects, only commit bin0 first.
        placed_ids = []
        remaining = []
        if max_bins > 1:
            base_bin_map = {rid: int(bin_idx) for (bin_idx, _, _, _, _, rid) in rects}
            for bin_idx, x, y, pw, ph, rid in rects:
                if bin_idx == 0:
                    placed_ids.append(rid)
            # Apply placements for bin0 only.
            placed_ids = _apply_rects([r for r in rects if r[0] == 0], 0, base_bin_map)
            rectpack_placed = len(placed_ids)
            order.extend(placed_ids)
            # Treat bin1 rects as "remaining" to be re-packed into gaps.
            remaining = [rid for (bin_idx, _, _, _, _, rid) in rects if bin_idx == 1]
        else:
            placed_ids = _apply_rects(rects, 0, base_bin_map)
            rectpack_placed = len(placed_ids)
            order.extend(placed_ids)
            remaining = [idx for idx in range(len(bboxes)) if idx not in set(placed_ids)]

        # Try to place remaining into right/bottom gaps of page 1 (original simple version).
        if remaining:
            def _recompute_used():
                umax_x = 0.0
                umax_y = 0.0
                for rid in placed_ids:
                    dx, dy, bw, bh, _rot = placements[rid]
                    info = rot_info[rid] if rid < len(rot_info) else {}
                    minx = float(info.get("minx", 0.0))
                    miny = float(info.get("miny", 0.0))
                    x = dx + minx
                    y = dy + miny
                    umax_x = max(umax_x, x + bw)
                    umax_y = max(umax_y, y + bh)
                return umax_x, umax_y

            def _rects_for_ids(ids: List[int]) -> List[Tuple[float, float, float, float]]:
                rects_out: List[Tuple[float, float, float, float]] = []
                for rid in ids:
                    dx, dy, bw, bh, _rot = placements[rid]
                    if bw <= 0 or bh <= 0:
                        continue
                    info = rot_info[rid] if rid < len(rot_info) else {}
                    minx = float(info.get("minx", 0.0))
                    miny = float(info.get("miny", 0.0))
                    x0 = dx + minx
                    y0 = dy + miny
                    rects_out.append((x0, y0, x0 + bw, y0 + bh))
                return rects_out

            def _overlaps_any(rects: List[Tuple[float, float, float, float]], cand: Tuple[float, float, float, float]) -> bool:
                cx0, cy0, cx1, cy1 = cand
                for x0, y0, x1, y1 in rects:
                    if cx1 <= x0 or cx0 >= x1 or cy1 <= y0 or cy0 >= y1:
                        continue
                    return True
                return False

            def _place_in_right_col(indices: List[int], gx: float, gw: float, gh: float) -> List[int]:
                if gw <= 0 or gh <= 0:
                    return []
                placed_local: List[int] = []
                existing_rects = _rects_for_ids(placed_ids)
                indices_sorted = sorted(indices, key=lambda i: bboxes[i][3] * bboxes[i][4], reverse=True)
                cur_y = 0.0
                for rid in indices_sorted:
                    best = None
                    for ang, (x0, y0, bw, bh) in [(0.0, bounds0[rid]), (90.0, bounds90[rid])]:
                        if bw > gw or bh > gh:
                            continue
                        nx = gx
                        ny = cur_y
                        if ny + bh > gh:
                            continue
                        score = (gw - (nx + bw - gx)) + (gh - (ny + bh))
                        if best is None or score > best[0]:
                            best = (score, nx, ny, x0, y0, bw, bh, ang)
                    if best is None:
                        continue
                    _score, nx, ny, x0, y0, bw, bh, ang = best
                    cand = (nx, ny, nx + bw, ny + bh)
                    if _overlaps_any(existing_rects, cand):
                        continue
                    dx = int(nx - x0)
                    dy = int(ny - y0)
                    placements[rid] = (dx, dy, int(bw), int(bh), False)
                    rot_info[rid]["bin"] = 0
                    if base_bin_map is not None:
                        rot_info[rid]["base_bin"] = int(base_bin_map.get(rid, 1))
                    rot_info[rid]["minx"] = float(x0)
                    rot_info[rid]["miny"] = float(y0)
                    rot_info[rid]["angle"] = float(ang)
                    placed_local.append(rid)
                    existing_rects.append(cand)
                    cur_y = ny + bh
                return placed_local

            def _place_in_bottom(indices: List[int], gx: float, gy: float, gw: float, gh: float) -> List[int]:
                if gw <= 0 or gh <= 0:
                    return []
                placed_local: List[int] = []
                existing_rects = _rects_for_ids(placed_ids)
                indices_sorted = sorted(indices, key=lambda i: bboxes[i][3] * bboxes[i][4], reverse=True)
                cur_x = gx
                cur_y = gy
                row_h = 0.0
                for rid in indices_sorted:
                    _idx, x0, y0, bw, bh = bboxes[rid]
                    if bw > gw or bh > gh:
                        continue
                    if cur_x + bw > gx + gw:
                        cur_x = gx
                        cur_y += row_h
                        row_h = 0.0
                    if cur_y + bh > gy + gh:
                        continue
                    dx = int(cur_x - x0)
                    dy = int(cur_y - y0)
                    cand = (cur_x, cur_y, cur_x + bw, cur_y + bh)
                    if _overlaps_any(existing_rects, cand):
                        continue
                    placements[rid] = (dx, dy, int(bw), int(bh), False)
                    rot_info[rid]["bin"] = 0
                    if base_bin_map is not None:
                        rot_info[rid]["base_bin"] = int(base_bin_map.get(rid, 1))
                    placed_local.append(rid)
                    existing_rects.append(cand)
                    cur_x += bw
                    row_h = max(row_h, bh)
                return placed_local

            # Snapshot base placements (before gap fill) for compaction targets.
            base_placed = placed_ids.copy()
            base_max_x, base_max_y = _recompute_used()
            used_max_x, used_max_y = base_max_x, base_max_y
            right_w = max(0.0, w - used_max_x)
            bottom_h = max(0.0, h - used_max_y)
            bottom_w = max(0.0, w - right_w)

            placed_right = _place_in_right_col(remaining, used_max_x, right_w, float(h))
            placed_ids.extend(placed_right)
            remaining = [idx for idx in remaining if idx not in set(placed_right)]
            if remaining:
                placed_bottom = _place_in_bottom(remaining, 0.0, used_max_y, bottom_w, bottom_h)
                placed_ids.extend(placed_bottom)
                remaining = [idx for idx in remaining if idx not in set(placed_bottom)]
            else:
                placed_bottom = []
            order.extend(placed_right)
            order.extend(placed_bottom)
            gap_right_added = len(placed_right)
            gap_bottom_added = len(placed_bottom)

            # Compact right-gap placements leftwards to touch old zones.
            if placed_right:
                base_rects = _rects_for_ids(base_placed)

                def _base_edge_x(y: float) -> float:
                    max_x = None
                    for x0, y0, x1, y1 in base_rects:
                        if y < y0 or y > y1:
                            continue
                        if max_x is None or x1 > max_x:
                            max_x = x1
                    return max_x if max_x is not None else base_max_x

                # For the first/last zone on right_path, use bottom-left/top-left only.
                first_rid = None
                first_y = None
                last_rid = None
                last_y = None
                for rid in placed_right:
                    dx, dy, bw, bh, _ = placements[rid]
                    info = rot_info[rid] if rid < len(rot_info) else {}
                    miny = float(info.get("miny", 0.0))
                    y0 = dy + miny
                    if first_y is None or y0 < first_y:
                        first_y = y0
                        first_rid = rid
                    if last_y is None or y0 > last_y:
                        last_y = y0
                        last_rid = rid

                for rid in placed_right:
                    dx, dy, bw, bh, rot = placements[rid]
                    info = rot_info[rid] if rid < len(rot_info) else {}
                    minx = float(info.get("minx", 0.0))
                    miny = float(info.get("miny", 0.0))
                    cur_x = dx + minx
                    cur_y = dy + miny
                    # Use bottom-left only for first zone, top-left only for last zone,
                    # and fallback to whichever edge exists if the other is missing.
                    top_edge = _base_edge_x(cur_y)
                    bot_edge = _base_edge_x(cur_y + bh)
                    if first_rid is not None and rid == first_rid:
                        target_edge = bot_edge
                    elif last_rid is not None and rid == last_rid:
                        target_edge = top_edge
                    else:
                        if top_edge is None:
                            target_edge = bot_edge
                        elif bot_edge is None:
                            target_edge = top_edge
                        else:
                            target_edge = max(top_edge, bot_edge)
                    if target_edge is None:
                        continue
                    # Align left edge of zone to right_path (top-left & bottom-left).
                    target_x = float(target_edge)
                    if cur_x > target_x:
                        shift = cur_x - target_x
                        placements[rid] = (int(dx - shift), dy, bw, bh, rot)

            # Compact bottom-gap placements upwards to touch old zones.
            for rid in placed_bottom:
                dx, dy, bw, bh, rot = placements[rid]
                info = rot_info[rid] if rid < len(rot_info) else {}
                miny = float(info.get("miny", 0.0))
                cur_y = dy + miny
                shift = cur_y - base_max_y
                if shift > 0:
                    placements[rid] = (dx, int(dy - shift), bw, bh, rot)

        if remaining:
            # Pack the still-remaining into page2.
            if try_heuristics:
                combos = [
                    (maxrects.MaxRectsBssf, rectpack.SORT_AREA),
                    (maxrects.MaxRectsBlsf, rectpack.SORT_AREA),
                    (maxrects.MaxRectsBaf, rectpack.SORT_AREA),
                    (maxrects.MaxRectsBl, rectpack.SORT_AREA),
                    (maxrects.MaxRectsBssf, rectpack.SORT_LSIDE),
                    (maxrects.MaxRectsBlsf, rectpack.SORT_LSIDE),
                    (maxrects.MaxRectsBaf, rectpack.SORT_LSIDE),
                    (maxrects.MaxRectsBssf, rectpack.SORT_SSIDE),
                    (maxrects.MaxRectsBlsf, rectpack.SORT_SSIDE),
                    (maxrects.MaxRectsBaf, rectpack.SORT_SSIDE),
                ]
                best2 = []
                for pack_algo, sort_algo in combos:
                    rects_try = _run_packer(pack_algo, sort_algo, remaining)
                    if len(rects_try) > len(best2):
                        best2 = rects_try
                    if len(best2) == len(remaining):
                        break
                rects2 = best2
            else:
                rects2 = _run_packer(None, None, remaining)
            if rects2:
                placed_ids2 = _apply_rects(rects2, 1)
                order.extend(placed_ids2)
    else:
        placed_ids = _apply_rects(rects, 0)
        order.extend(placed_ids)
        rectpack_placed = len(placed_ids)
        remaining = []

    PACK_DEBUG_STATS = {
        "rectpack_placed": rectpack_placed,
        "gap_right_added": gap_right_added,
        "gap_bottom_added": gap_bottom_added,
        "total_placed": len([p for p in placements if p[2] > 0 and p[3] > 0]),
        "remaining_after_gap": len(remaining) if "remaining" in locals() else 0,
    }
    return placements, order, rot_info


def pack_regions_nested(
    polys: List[List[Tuple[float, float]]],
    canvas: Tuple[int, int],
    angle_step: float = 5.0,
    grid_step: float = 5.0,
    fixed_centers: List[Tuple[float, float]] | None = None,
    max_bins: int = 2,
) -> Tuple[List[Tuple[int, int, int, int, bool]], List[int], List[Dict[str, float]]]:
    """True polygon-aware nesting (no rectpack boxes)."""
    w, h = canvas
    pad = float(config.PADDING)
    x_min = float(config.PACK_MARGIN_X)
    y_min = float(config.PACK_MARGIN_Y)
    x_max = float(w - config.PACK_MARGIN_X)
    y_max = float(h - config.PACK_MARGIN_Y)
    step = max(0.5, float(grid_step))
    angle_step = max(1.0, float(angle_step))

    n = len(polys)
    placements: List[Tuple[int, int, int, int, bool]] = [(-1, -1, 0, 0, False)] * n
    rot_info: List[Dict[str, float]] = [{"angle": 0.0, "cx": 0.0, "cy": 0.0, "minx": 0.0, "miny": 0.0} for _ in range(n)]
    order: List[int] = []

    # Build stable angle candidates
    angles: List[float] = [0.0]
    a = angle_step
    while a <= 180.0 + 1e-6:
        aval = float(round(a, 6))
        if aval not in angles:
            angles.append(aval)
        a += angle_step
    if 90.0 not in angles:
        angles.append(90.0)
    angles = sorted(set(angles))

    # Prepare sort by area desc (big first)
    areas = []
    for i, pts in enumerate(polys):
        try:
            pg = Polygon(pts)
            areas.append((i, abs(float(pg.area))))
        except Exception:
            areas.append((i, 0.0))
    place_ids = [i for i, _ in sorted(areas, key=lambda t: t[1], reverse=True)]

    # Per-bin occupied collision polys with cached bboxes
    occ: Dict[int, List[Tuple[Tuple[float, float, float, float], BaseGeometry]]] = {b: [] for b in range(max_bins)}

    def _bbox_overlap(a0, b0) -> bool:
        return not (a0[2] <= b0[0] or a0[0] >= b0[2] or a0[3] <= b0[1] or a0[1] >= b0[3])

    for rid in place_ids:
        pts = polys[rid]
        if not pts or len(pts) < 3:
            continue
        try:
            base_poly = Polygon(pts)
            if base_poly.is_empty:
                continue
        except Exception:
            continue

        if fixed_centers is not None and rid < len(fixed_centers):
            cx, cy = fixed_centers[rid]
        else:
            c = base_poly.centroid
            cx, cy = float(c.x), float(c.y)

        best = None
        for bin_idx in range(max_bins):
            bin_occ = occ[bin_idx]
            for ang in angles:
                rpts = _rotate_pts(pts, ang, cx, cy)
                try:
                    rpoly = Polygon(rpts)
                    if rpoly.is_empty:
                        continue
                except Exception:
                    continue
                collision_poly = rpoly.buffer(pad * 0.5, join_style=2) if pad > 0 else rpoly
                cminx, cminy, cmaxx, cmaxy = collision_poly.bounds
                cw = cmaxx - cminx
                ch = cmaxy - cminy
                if cw <= 0 or ch <= 0:
                    continue
                if cw > (x_max - x_min) or ch > (y_max - y_min):
                    continue

                y = y_min
                while y <= (y_max - ch + 1e-6):
                    x = x_min
                    while x <= (x_max - cw + 1e-6):
                        tx = x - cminx
                        ty = y - cminy
                        cand = _stranslate(collision_poly, xoff=tx, yoff=ty)
                        bb = cand.bounds
                        collide = False
                        for obb, op in bin_occ:
                            if not _bbox_overlap(bb, obb):
                                continue
                            try:
                                if cand.intersects(op) and not cand.touches(op):
                                    collide = True
                                    break
                            except Exception:
                                continue
                        if not collide:
                            # Bottom-left score
                            score = y * float(w) + x
                            if best is None or score < best["score"]:
                                rminx, rminy, rmaxx, rmaxy = rpoly.bounds
                                best = {
                                    "rid": rid,
                                    "bin": bin_idx,
                                    "score": score,
                                    "tx": tx,
                                    "ty": ty,
                                    "ang": float(ang),
                                    "cx": float(cx),
                                    "cy": float(cy),
                                    "minx": float(rminx),
                                    "miny": float(rminy),
                                    "bw": int(ceil((rmaxx - rminx) + pad * 2)),
                                    "bh": int(ceil((rmaxy - rminy) + pad * 2)),
                                    "occ_poly": cand,
                                }
                            # Fast early-exit: first valid spot for current angle/bin
                            break
                        x += step
                    if best is not None and best["rid"] == rid:
                        break
                    y += step
                if best is not None and best["rid"] == rid:
                    break
            if best is not None and best["rid"] == rid:
                break

        if best is None:
            continue

        dx = int(round(best["tx"]))
        dy = int(round(best["ty"]))
        placements[rid] = (dx, dy, int(best["bw"]), int(best["bh"]), False)
        rot_info[rid] = {
            "angle": float(best["ang"]),
            "cx": float(best["cx"]),
            "cy": float(best["cy"]),
            "minx": float(best["minx"]),
            "miny": float(best["miny"]),
            "bin": int(best["bin"]),
        }
        occ[int(best["bin"])].append((best["occ_poly"].bounds, best["occ_poly"]))
        order.append(rid)

    return placements, order, rot_info


def pack_regions_raster_fast(
    polys: List[List[Tuple[float, float]]],
    canvas: Tuple[int, int],
    fixed_centers: List[Tuple[float, float]] | None = None,
    grid_step: float = 4.0,
    rotations: List[float] | None = None,
    search_stride: int = 1,
    safety_padding: float = 3.0,
    place_ids: List[int] | None = None,
) -> Tuple[List[Tuple[int, int, int, int, bool]], List[int], List[Dict[str, float]]]:
    """TS-style raster occupancy packer (single page), optimized for speed."""
    global PACK_DEBUG_STATS
    w, h = canvas
    safety = max(0.0, float(safety_padding))
    edge_safe = int(ceil(safety))
    x_min = int(config.PACK_MARGIN_X) + edge_safe
    y_min = int(config.PACK_MARGIN_Y) + edge_safe
    x_max = int(w - config.PACK_MARGIN_X) - edge_safe
    y_max = int(h - config.PACK_MARGIN_Y) - edge_safe
    cell = max(1, int(round(float(grid_step))))
    bw = max(1, int(ceil((x_max - x_min) / float(cell))))
    bh = max(1, int(ceil((y_max - y_min) / float(cell))))
    step = 1
    # Ultra-fast mode: ignore geometric padding in raster mask generation.
    pad = 0.0

    if rotations is None:
        # Cardinal rotations are a good speed/quality tradeoff.
        rotations = [0.0, 90.0, 180.0, 270.0]

    n = len(polys)
    if place_ids is None:
        place_ids_norm = list(range(n))
    else:
        seen: set[int] = set()
        place_ids_norm: List[int] = []
        for rid in place_ids:
            rr = int(rid)
            if rr < 0 or rr >= n or rr in seen:
                continue
            seen.add(rr)
            place_ids_norm.append(rr)

    placements: List[Tuple[int, int, int, int, bool]] = [(-1, -1, 0, 0, False)] * n
    rot_info: List[Dict[str, float]] = [{"angle": 0.0, "cx": 0.0, "cy": 0.0, "minx": 0.0, "miny": 0.0, "bin": -1} for _ in range(n)]
    order: List[int] = []

    grid = np.zeros((bh, bw), dtype=np.uint8)

    def _dilate_mask(mask: np.ndarray, radius_cells: int) -> np.ndarray:
        if radius_cells <= 0 or mask.size == 0:
            return mask
        out = mask.copy()
        for _ in range(radius_cells):
            p = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=0)
            out = (
                p[1:-1, 1:-1]
                | p[:-2, 1:-1]
                | p[2:, 1:-1]
                | p[1:-1, :-2]
                | p[1:-1, 2:]
                | p[:-2, :-2]
                | p[:-2, 2:]
                | p[2:, :-2]
                | p[2:, 2:]
            ).astype(np.uint8)
        return out

    def _rasterize_poly(pts: List[Tuple[float, float]], minx: float, miny: float, mw: int, mh: int) -> np.ndarray:
        if mw <= 0 or mh <= 0:
            return np.zeros((0, 0), dtype=np.uint8)
        img = Image.new("1", (mw, mh), 0)
        draw = ImageDraw.Draw(img)
        pix = [((float(px - minx) / float(cell)), (float(py - miny) / float(cell))) for px, py in pts]
        draw.polygon(pix, fill=1, outline=1)
        return np.array(img, dtype=np.uint8)

    # Precompute masks for each shape x rotation
    masks: Dict[Tuple[int, float], Tuple[np.ndarray, float, float, int, int]] = {}
    centers: Dict[int, Tuple[float, float]] = {}
    areas: List[Tuple[int, float]] = []
    for rid in place_ids_norm:
        pts = polys[rid] if rid < len(polys) else []
        if not pts or len(pts) < 3:
            areas.append((rid, 0.0))
            continue
        try:
            pg0 = Polygon(pts)
            areas.append((rid, abs(float(pg0.area))))
        except Exception:
            areas.append((rid, 0.0))
            continue
        if fixed_centers is not None and rid < len(fixed_centers):
            cx, cy = fixed_centers[rid]
        else:
            c = pg0.centroid
            cx, cy = float(c.x), float(c.y)
        centers[rid] = (float(cx), float(cy))
        for ang in rotations:
            rpts = _rotate_pts(pts, float(ang), cx, cy)
            try:
                xs = [p[0] for p in rpts]
                ys = [p[1] for p in rpts]
                minx, miny = float(min(xs)), float(min(ys))
                maxx, maxy = float(max(xs)), float(max(ys))
                if pad > 0:
                    minx -= pad
                    miny -= pad
                    maxx += pad
                    maxy += pad
                mw = int(ceil((maxx - minx) / float(cell)))
                mh = int(ceil((maxy - miny) / float(cell)))
                if mw <= 0 or mh <= 0:
                    continue
                if mw > bw or mh > bh:
                    continue
                m = _rasterize_poly(rpts, minx, miny, mw, mh)
                if m.size <= 0 or not np.any(m):
                    continue
                safety_cells = int(ceil(max(0.0, float(safety_padding)) / float(cell)))
                if safety_cells > 0:
                    m = _dilate_mask(m, safety_cells)
                # Store both raster mask size (mw/mh in cells) and real bbox size (ww/hh in world units).
                ww = float(maxx - minx)
                hh = float(maxy - miny)
                masks[(rid, float(ang))] = (m, float(minx), float(miny), mw, mh, ww, hh)
            except Exception:
                continue

    # Big first tends to reduce fragmentation. Caller can override order.
    if place_ids is None:
        place_ids = [rid for rid, _a in sorted(areas, key=lambda t: t[1], reverse=True)]
    else:
        # keep valid and unique ids only
        seen: set[int] = set()
        normalized: List[int] = []
        for rid in place_ids:
            rr = int(rid)
            if rr < 0 or rr >= n or rr in seen:
                continue
            seen.add(rr)
            normalized.append(rr)
        place_ids = normalized
    placed_count = 0

    for rid in place_ids:
        best = None
        rot_candidates = [
            float(ang)
            for ang in rotations
            if (rid, float(ang)) in masks
        ]
        # Prefer "horizontal" orientation first (smaller bbox height).
        rot_candidates.sort(key=lambda a: (masks[(rid, a)][4], -masks[(rid, a)][3], a))
        for ang in rot_candidates:
            key = (rid, float(ang))
            if key not in masks:
                continue
            mask, minx, miny, mw, mh, ww, hh = masks[key]
            lim_y = bh - mh
            lim_x = bw - mw
            y = 0
            while y <= lim_y:
                x = 0
                while x <= lim_x:
                    # Cheap early skip like TS.
                    if grid[y, x] == 1:
                        x += step
                        continue
                    reg = grid[y : y + mh, x : x + mw]
                    if not np.any(reg & mask):
                        # TS-style top-left first, with tie-break toward flatter (horizontal) bbox.
                        score = (y, x, mh, -mw)
                        if best is None or score < best[0]:
                            best = (score, x, y, ang, minx, miny, mw, mh, ww, hh, mask)
                    x += max(step, int(search_stride))
                y += max(step, int(search_stride))

        if best is None:
            continue
        _score, x, y, ang, minx, miny, mw, mh, ww, hh, mask = best
        grid[y : y + mh, x : x + mw] |= mask
        dx = int(round((x_min + x * cell) - minx))
        dy = int(round((y_min + y * cell) - miny))
        # placements must be in world units (not raster cells) so downstream fit/bounds stay correct.
        placements[rid] = (dx, dy, int(ceil(max(0.0, ww))), int(ceil(max(0.0, hh))), False)
        rot_info[rid] = {
            "angle": float(ang),
            "cx": float(centers.get(rid, (0.0, 0.0))[0]),
            "cy": float(centers.get(rid, (0.0, 0.0))[1]),
            "minx": float(minx),
            "miny": float(miny),
            "bbox_w": float(max(0.0, ww)),
            "bbox_h": float(max(0.0, hh)),
            "bin": 0,
        }
        order.append(rid)
        placed_count += 1

    PACK_DEBUG_STATS = {
        "rectpack_placed": 0,
        "gap_right_added": 0,
        "gap_bottom_added": 0,
        "remaining_after_gap": max(0, n - placed_count),
        "total_placed": placed_count,
    }
    return placements, order, rot_info


def raster_overlap_report(
    zone_polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[int, int, int, int, bool]],
    rot_info: List[Dict[str, float]],
    canvas: Tuple[int, int],
    cell: int = 1,
) -> Dict[str, object]:
    """Fast raster collision check for final placed polygons."""
    if not zone_polys or not placements or not rot_info:
        return {"count": 0, "pairs": []}
    w, h = int(canvas[0]), int(canvas[1])
    c = max(1, int(cell))
    gw = max(1, int(ceil(w / float(c))))
    gh = max(1, int(ceil(h / float(c))))
    owners: Dict[int, np.ndarray] = {}
    pairs: set[Tuple[int, int]] = set()

    def _owner_for_bin(bin_idx: int) -> np.ndarray:
        arr = owners.get(bin_idx)
        if arr is None:
            arr = np.full((gh, gw), -1, dtype=np.int32)
            owners[bin_idx] = arr
        return arr

    for rid in range(min(len(zone_polys), len(placements), len(rot_info))):
        pts = zone_polys[rid]
        if not pts or len(pts) < 3:
            continue
        dx, dy, bw, bh, _ = placements[rid]
        if bw <= 0 or bh <= 0:
            continue
        info = rot_info[rid]
        ang = float(info.get("angle", 0.0))
        cx = float(info.get("cx", 0.0))
        cy = float(info.get("cy", 0.0))
        bin_idx = int(info.get("bin", 0))
        tpts = [(p[0] + dx, p[1] + dy) for p in _rotate_pts(pts, ang, cx, cy)]
        xs = [p[0] for p in tpts]
        ys = [p[1] for p in tpts]
        minx = max(0, int(np.floor(min(xs) / float(c))))
        miny = max(0, int(np.floor(min(ys) / float(c))))
        maxx = min(gw, int(np.ceil(max(xs) / float(c))))
        maxy = min(gh, int(np.ceil(max(ys) / float(c))))
        bwc = maxx - minx
        bhc = maxy - miny
        if bwc <= 0 or bhc <= 0:
            continue
        pix = [((float(px) / float(c)) - minx, (float(py) / float(c)) - miny) for px, py in tpts]
        img = Image.new("1", (bwc, bhc), 0)
        draw = ImageDraw.Draw(img)
        try:
            draw.polygon(pix, fill=1, outline=1)
        except Exception:
            continue
        mask = np.array(img, dtype=np.uint8) > 0
        if not np.any(mask):
            continue
        owner = _owner_for_bin(bin_idx)
        sub = owner[miny:maxy, minx:maxx]
        hit = sub[mask]
        if hit.size:
            touched = np.unique(hit[hit >= 0])
            for oid in touched:
                a = int(min(oid, rid))
                b = int(max(oid, rid))
                if a != b:
                    pairs.add((a, b))
        sub[mask] = rid

    pair_list = [[int(a), int(b)] for (a, b) in sorted(pairs)]
    return {"count": len(pair_list), "pairs": pair_list}


def repack_page2_into_page1_nested(
    polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[int, int, int, int, bool]],
    rot_info: List[Dict[str, float]],
    canvas: Tuple[int, int],
    angle_step: float = 5.0,
    grid_step: float = 5.0,
    max_moves: int | None = None,
) -> Tuple[List[Tuple[int, int, int, int, bool]], List[Dict[str, float]], int]:
    """Try moving bin=1 zones back into bin=0 with polygon collision checks."""
    if not polys or not placements or not rot_info:
        return placements, rot_info, 0

    w, h = canvas
    pad = float(config.PADDING)
    x_min = float(config.PACK_MARGIN_X)
    y_min = float(config.PACK_MARGIN_Y)
    x_max = float(w - config.PACK_MARGIN_X)
    y_max = float(h - config.PACK_MARGIN_Y)
    step = max(0.5, float(grid_step))
    angle_step = max(1.0, float(angle_step))

    out_p = list(placements)
    out_r = [dict(r or {}) for r in rot_info]

    def _bbox_overlap(a0, b0) -> bool:
        return not (a0[2] <= b0[0] or a0[0] >= b0[2] or a0[3] <= b0[1] or a0[1] >= b0[3])

    def _zone_poly_at(rid: int, ang: float, dx: float, dy: float):
        if rid >= len(polys) or not polys[rid] or len(polys[rid]) < 3:
            return None
        info = out_r[rid] if rid < len(out_r) else {}
        cx = float(info.get("cx", 0.0))
        cy = float(info.get("cy", 0.0))
        rpts = _rotate_pts(polys[rid], ang, cx, cy)
        tpts = [(p[0] + dx, p[1] + dy) for p in rpts]
        try:
            pg = Polygon(tpts)
            if pg.is_empty:
                return None
            return pg.buffer(pad * 0.5, join_style=2) if pad > 0 else pg
        except Exception:
            return None

    occ: List[Tuple[Tuple[float, float, float, float], BaseGeometry]] = []
    for rid, pl in enumerate(out_p):
        if rid >= len(out_r) or rid >= len(polys):
            continue
        dx, dy, bw, bh, _ = pl
        if bw <= 0 or bh <= 0:
            continue
        if int(out_r[rid].get("bin", 0)) != 0:
            continue
        ang = float(out_r[rid].get("angle", 0.0))
        pg = _zone_poly_at(rid, ang, float(dx), float(dy))
        if pg is None:
            continue
        occ.append((pg.bounds, pg))

    # Move smaller leftovers first to increase chance of filling holes.
    movers: List[Tuple[int, float]] = []
    for rid, pl in enumerate(out_p):
        if rid >= len(out_r) or rid >= len(polys):
            continue
        dx, dy, bw, bh, _ = pl
        if bw <= 0 or bh <= 0:
            continue
        if int(out_r[rid].get("bin", 0)) != 1:
            continue
        area = float(max(1, bw) * max(1, bh))
        movers.append((rid, area))
    movers.sort(key=lambda t: t[1])
    if max_moves is not None and max_moves > 0:
        movers = movers[:max_moves]

    # Build stable angle candidates.
    base_angles: List[float] = [0.0]
    a = angle_step
    while a <= 180.0 + 1e-6:
        aval = float(round(a, 6))
        if aval not in base_angles:
            base_angles.append(aval)
        a += angle_step
    if 90.0 not in base_angles:
        base_angles.append(90.0)
    base_angles = sorted(set(base_angles))

    moved = 0
    for rid, _a in movers:
        info = out_r[rid] if rid < len(out_r) else {}
        cur_ang = float(info.get("angle", 0.0))
        angles = [cur_ang] + [ang for ang in base_angles if abs(ang - cur_ang) > 1e-6]
        placed = False
        for ang in angles:
            try:
                rpts = _rotate_pts(polys[rid], ang, float(info.get("cx", 0.0)), float(info.get("cy", 0.0)))
                rpoly = Polygon(rpts)
                if rpoly.is_empty:
                    continue
            except Exception:
                continue
            collision_poly = rpoly.buffer(pad * 0.5, join_style=2) if pad > 0 else rpoly
            cminx, cminy, cmaxx, cmaxy = collision_poly.bounds
            cw = cmaxx - cminx
            ch = cmaxy - cminy
            if cw <= 0 or ch <= 0:
                continue
            if cw > (x_max - x_min) or ch > (y_max - y_min):
                continue

            y = y_min
            while y <= (y_max - ch + 1e-6):
                x = x_min
                while x <= (x_max - cw + 1e-6):
                    tx = x - cminx
                    ty = y - cminy
                    cand = _stranslate(collision_poly, xoff=tx, yoff=ty)
                    bb = cand.bounds
                    collide = False
                    for obb, op in occ:
                        if not _bbox_overlap(bb, obb):
                            continue
                        try:
                            if cand.intersects(op) and not cand.touches(op):
                                collide = True
                                break
                        except Exception:
                            continue
                    if not collide:
                        rminx, rminy, rmaxx, rmaxy = rpoly.bounds
                        out_p[rid] = (
                            int(round(tx)),
                            int(round(ty)),
                            int(ceil((rmaxx - rminx) + pad * 2)),
                            int(ceil((rmaxy - rminy) + pad * 2)),
                            False,
                        )
                        out_r[rid]["angle"] = float(ang)
                        out_r[rid]["minx"] = float(rminx)
                        out_r[rid]["miny"] = float(rminy)
                        out_r[rid]["bin"] = 0
                        occ.append((cand.bounds, cand))
                        moved += 1
                        placed = True
                        break
                    x += step
                if placed:
                    break
                y += step
            if placed:
                break

    return out_p, out_r, moved


def compact_nesting_polygons(
    polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[int, int, int, int, bool]],
    rot_info: List[Dict[str, float]],
    canvas: Tuple[int, int],
    step: float = 1.0,
    passes: int = 2,
) -> List[Tuple[int, int, int, int, bool]]:
    """Local polygon compaction pass to reduce visual gaps after box packing."""
    w, h = canvas
    step = max(0.25, float(step))
    out = list(placements)

    def _poly_for(rid: int) -> BaseGeometry | None:
        if rid < 0 or rid >= len(polys) or rid >= len(out):
            return None
        pts = polys[rid]
        if not pts:
            return None
        dx, dy, bw, bh, _ = out[rid]
        if bw <= 0 or bh <= 0:
            return None
        info = rot_info[rid] if rid < len(rot_info) else {}
        ang = float(info.get("angle", 0.0))
        cx = float(info.get("cx", 0.0))
        cy = float(info.get("cy", 0.0))
        tpts = [(p[0] + dx, p[1] + dy) for p in _rotate_pts(pts, ang, cx, cy)]
        try:
            pg = Polygon(tpts)
            if not pg.is_valid:
                pg = make_valid(pg)
            return pg
        except Exception:
            return None

    def _fits(poly: BaseGeometry, rid: int, xdir: float, ydir: float, polys_now: List[BaseGeometry | None]) -> bool:
        cand = _stranslate(poly, xoff=xdir, yoff=ydir)
        if cand.is_empty:
            return False
        minx, miny, maxx, maxy = cand.bounds
        if minx < 0 or miny < 0 or maxx > w or maxy > h:
            return False
        bin_i = int((rot_info[rid] if rid < len(rot_info) else {}).get("bin", 0))
        for j, other in enumerate(polys_now):
            if j == rid or other is None:
                continue
            if out[j][2] <= 0 or out[j][3] <= 0:
                continue
            bin_j = int((rot_info[j] if j < len(rot_info) else {}).get("bin", 0))
            if bin_i != bin_j:
                continue
            try:
                if cand.intersects(other) and not cand.touches(other):
                    return False
            except Exception:
                continue
        return True

    transformed: List[BaseGeometry | None] = [_poly_for(i) for i in range(len(out))]
    ids = [i for i, p in enumerate(transformed) if p is not None]

    for _ in range(max(1, int(passes))):
        moved_any = False
        ids_sorted = sorted(
            ids,
            key=lambda i: (
                float((transformed[i].bounds[1] if transformed[i] is not None else 0.0)),
                float((transformed[i].bounds[0] if transformed[i] is not None else 0.0)),
            ),
        )
        for rid in ids_sorted:
            cur = transformed[rid]
            if cur is None:
                continue
            moved = True
            while moved:
                moved = False
                if _fits(cur, rid, -step, 0.0, transformed):
                    cur = _stranslate(cur, xoff=-step, yoff=0.0)
                    dx, dy, bw, bh, rot = out[rid]
                    out[rid] = (int(round(dx - step)), dy, bw, bh, rot)
                    moved = True
                    moved_any = True
            moved = True
            while moved:
                moved = False
                if _fits(cur, rid, 0.0, -step, transformed):
                    cur = _stranslate(cur, xoff=0.0, yoff=-step)
                    dx, dy, bw, bh, rot = out[rid]
                    out[rid] = (dx, int(round(dy - step)), bw, bh, rot)
                    moved = True
                    moved_any = True
            transformed[rid] = cur
        if not moved_any:
            break

    return out


def _build_packed_order_by_bin(rects: List[Tuple[int, int, int, int, int, int]]) -> Dict[int, List[int]]:
    order: Dict[int, List[int]] = {}
    for bin_idx, _, _, _, _, rid in rects:
        order.setdefault(bin_idx, []).append(rid)
    return order


def write_pack_log(
    polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[int, int, int, int, bool]],
    rot_info: List[Dict[str, float]],
    zone_label_map: Dict[int, int] | None,
    out_path,
    canvas: Tuple[int, int],
) -> None:
    w, h = canvas
    visible = 0
    overflow = 0
    placed = 0
    unplaced = 0
    placed_area = 0
    lines = ["packed_regions"]
    for rid, (dx, dy, bw, bh, rot) in enumerate(placements):
        info = rot_info[rid] if rid < len(rot_info) else {}
        minx = float(info.get("minx", 0.0))
        miny = float(info.get("miny", 0.0))
        x0 = dx + minx
        y0 = dy + miny
        x1 = x0 + bw
        y1 = y0 + bh
        if bw <= 0 or bh <= 0:
            unplaced += 1
        else:
            placed += 1
            placed_area += int(bw) * int(bh)
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            overflow += 1
        else:
            visible += 1
        alias = zone_label_map.get(rid) if zone_label_map else None
        alias_str = f" alias={alias}" if alias is not None else ""
        lines.append(
            f"region={rid} dx={dx} dy={dy} x={x0:.2f} y={y0:.2f} w={bw} h={bh} rot={rot}{alias_str}"
        )
    lines.append(f"packed_visible={visible} overflow={overflow}")
    lines.append(f"packed_placed={placed} packed_unplaced={unplaced} placed_area={placed_area} bin_area={int(w*h)}")
    if PACK_DEBUG_STATS:
        lines.append(
            "gap_fill rectpack_placed="
            f"{PACK_DEBUG_STATS.get('rectpack_placed', 0)} "
            f"gap_right_added={PACK_DEBUG_STATS.get('gap_right_added', 0)} "
            f"gap_bottom_added={PACK_DEBUG_STATS.get('gap_bottom_added', 0)} "
            f"remaining_after_gap={PACK_DEBUG_STATS.get('remaining_after_gap', 0)} "
            f"total_placed={PACK_DEBUG_STATS.get('total_placed', 0)}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_pack_bbox_svg(
    placements: List[Tuple[int, int, int, int, bool]],
    rot_info: List[Dict[str, float]],
    canvas: Tuple[int, int],
    out_path: Path,
    *,
    zone_label_map: Dict[int, int] | None = None,
    packed_order: List[int] | None = None,
) -> None:
    w, h = canvas
    visible = 0
    used_max_x = 0.0
    used_max_y = 0.0
    base_max_x = None
    base_rects: List[Tuple[float, float, float, float]] = []
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="none" stroke="#ffffff" stroke-width="1"/>',
        '<g id="bboxes">',
    ]
    seq = 0
    indices = packed_order if packed_order is not None else list(range(len(placements)))
    for rid in indices:
        if rid < 0 or rid >= len(placements):
            continue
        dx, dy, bw, bh, _rot = placements[rid]
        if rid >= len(rot_info):
            continue
        if bw <= 0 or bh <= 0:
            continue
        info = rot_info[rid]
        x0 = float(info.get("minx", 0.0))
        y0 = float(info.get("miny", 0.0))
        x = float(dx) + x0
        y = float(dy) + y0
        parts.append(
            f'<rect x="{x:.3f}" y="{y:.3f}" width="{float(bw):.3f}" height="{float(bh):.3f}" '
            f'fill="none" stroke="#00ff7f" stroke-width="1"/>'
        )
        if int(info.get("base_bin", 0)) == 0:
            bx = x + bw
            base_max_x = bx if base_max_x is None else max(base_max_x, bx)
            base_rects.append((x, y, x + bw, y + bh))
        if x + bw > 0 and y + bh > 0 and x < w and y < h:
            visible += 1
            used_max_x = max(used_max_x, x + bw)
            used_max_y = max(used_max_y, y + bh)
        label = seq if packed_order is not None else (zone_label_map.get(rid, rid) if zone_label_map else rid)
        parts.append(
            f'<text x="{x + 2:.3f}" y="{y + 12:.3f}" fill="#00ff7f" font-size="10">{label}</text>'
        )
        seq += 1
    # Log remaining right/bottom margins based on used extents.
    right_w = max(0.0, w - used_max_x)
    bottom_h = max(0.0, h - used_max_y)
    if right_w > 0:
        parts.append(
            f'<rect x="{used_max_x:.3f}" y="0" width="{right_w:.3f}" height="{h:.3f}" '
            f'fill="none" stroke="#ff5252" stroke-width="1" stroke-dasharray="6,4"/>'
        )
        parts.append(
            f'<text x="{used_max_x + 6:.3f}" y="16" fill="#ff5252" font-size="12">right_gap</text>'
        )
    if bottom_h > 0:
        bottom_w = max(0.0, w - right_w)
        parts.append(
            f'<rect x="0" y="{used_max_y:.3f}" width="{bottom_w:.3f}" height="{bottom_h:.3f}" '
            f'fill="none" stroke="#ff5252" stroke-width="1" stroke-dasharray="6,4"/>'
        )
        parts.append(
            f'<text x="6" y="{used_max_y + 16:.3f}" fill="#ff5252" font-size="12">bottom_gap</text>'
        )
    parts.append(
        f'<text x="6" y="14" fill="#ffffff" font-size="12">visible={visible}</text>'
    )
    if base_max_x is not None and base_rects:
        step = 2
        pts = []
        last_x = None
        min_x = None
        y = 0
        while y <= h:
            max_x = None
            for x0, y0, x1, y1 in base_rects:
                if y < y0 or y > y1:
                    continue
                if max_x is None or x1 > max_x:
                    max_x = x1
            if max_x is None:
                y += step
                continue
            last_x = max_x
            min_x = max_x if min_x is None else min(min_x, max_x)
            pts.append((max_x, y))
            y += step
        if pts:
            d = "M " + " L ".join(f"{px:.3f} {py:.3f}" for px, py in pts)
            parts.append(
                f'<path d="{d}" fill="none" stroke="#ff9800" stroke-width="1" stroke-dasharray="6,4"/>'
            )
            parts.append(
                f'<text x="{pts[0][0] + 4:.3f}" y="{min(16, h-4):.3f}" fill="#ff9800" font-size="12">base_right_edge</text>'
            )
        # Bottom edge polyline (scan along X).
        pts_b = []
        last_y = None
        min_y = None
        x = 0
        while x <= w:
            max_y = None
            for x0, y0, x1, y1 in base_rects:
                if x < x0 or x > x1:
                    continue
                if max_y is None or y1 > max_y:
                    max_y = y1
            if max_y is None:
                if last_y is None:
                    max_y = 0.0
                else:
                    max_y = last_y
            else:
                last_y = max_y
            min_y = max_y if min_y is None else min(min_y, max_y)
            pts_b.append((x, max_y))
            x += step
        if pts_b:
            if min_y is not None:
                pts_b[0] = (pts_b[0][0], min_y)
                if len(pts_b) > 1:
                    pts_b[1] = (pts_b[1][0], min_y)
            d = "M " + " L ".join(f"{px:.3f} {py:.3f}" for px, py in pts_b)
            parts.append(
                f'<path d="{d}" fill="none" stroke="#ff9800" stroke-width="1" stroke-dasharray="6,4"/>'
            )
            parts.append(
                f'<text x="{min(4, w-4):.3f}" y="{pts_b[0][1] + 14:.3f}" fill="#ff9800" font-size="12">base_bottom_edge</text>'
            )
    parts.append("</g></svg>")
    out_path.write_text("".join(parts), encoding="utf-8")


def write_pack_bbox_vs_poly_svg(
    zone_polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[int, int, int, int, bool]],
    rot_info: List[Dict[str, float]],
    canvas: Tuple[int, int],
    out_path: Path,
    *,
    zone_label_map: Dict[int, int] | None = None,
    packed_order: List[int] | None = None,
) -> None:
    w, h = canvas
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="none" stroke="#ffffff" stroke-width="1"/>',
        '<g id="bboxes">',
    ]
    indices = packed_order if packed_order is not None else list(range(len(placements)))
    for rid in indices:
        if rid < 0 or rid >= len(placements) or rid >= len(rot_info):
            continue
        dx, dy, bw, bh, _rot = placements[rid]
        if dx < 0 or dy < 0 or bw <= 0 or bh <= 0:
            continue
        info = rot_info[rid]
        x0 = float(info.get("minx", 0.0))
        y0 = float(info.get("miny", 0.0))
        x = float(dx) + x0
        y = float(dy) + y0
        parts.append(
            f'<rect x="{x:.3f}" y="{y:.3f}" width="{float(bw):.3f}" height="{float(bh):.3f}" '
            f'fill="none" stroke="#00ff7f" stroke-width="1"/>'
        )
    parts.append("</g>")
    parts.append('<g id="polys">')
    for rid, pts in enumerate(zone_polys):
        if rid >= len(placements) or rid >= len(rot_info):
            continue
        if packed_order is not None and rid not in set(indices):
            continue
        dx, dy, bw, bh, _ = placements[rid]
        if dx < 0 or dy < 0 or bw <= 0 or bh <= 0:
            continue
        info = rot_info[rid]
        ang = float(info.get("angle", 0.0))
        cx = float(info.get("cx", 0.0))
        cy = float(info.get("cy", 0.0))
        rpts = _rotate_pts(pts, ang, cx, cy)
        tpts = [(p[0] + dx, p[1] + dy) for p in rpts]
        if not tpts:
            continue
        d = "M " + " L ".join(f"{p[0]:.3f} {p[1]:.3f}" for p in tpts) + " Z"
        parts.append(f'<path d="{d}" fill="none" stroke="#ff3b30" stroke-width="1"/>')
        label = zone_label_map.get(rid, rid) if zone_label_map else rid
        lx = tpts[0][0]
        ly = tpts[0][1]
        parts.append(f'<text x="{lx + 2:.3f}" y="{ly + 12:.3f}" fill="#ff3b30" font-size="10">{label}</text>')
    parts.append("</g></svg>")
    out_path.write_text("".join(parts), encoding="utf-8")


def build_bleed(group_mask: np.ndarray, canvas_fill: np.ndarray, border_thick: int) -> Tuple[np.ndarray, np.ndarray]:
    # Raster bleed disabled (cv2 removed).
    return group_mask, canvas_fill


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
    *,
    draw_scale: float | None = None,
    out_path: Path | None = None,
) -> None:
    # Raster output disabled (cv2 removed).
    return


def write_pack_svg(
    polys: List[List[Tuple[float, float]]],
    zone_id: List[int],
    zone_order: List[int],
    zone_polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[int, int, int, int, bool]],
    canvas: Tuple[int, int],
    colors: List[Tuple[int, int, int]],
    rot_info: List[Dict[str, float]],
    *,
    placement_bin: List[int] | None = None,
    placement_bin_by_zid: Dict[int, int] | None = None,
    page_idx: int | None = None,
    out_path: Path | None = None,
    include_bleed: bool = True,
) -> None:
    w, h = canvas
    zone_shift: Dict[int, Tuple[float, float]] = {}
    zone_rot: Dict[int, float] = {}
    zone_center: Dict[int, Tuple[float, float]] = {}
    bleed_canvas = float(config.PACK_BLEED)
    for idx, zid in enumerate(zone_order):
        if idx >= len(placements):
            continue
        dx, dy, _, _, _ = placements[idx]
        zone_shift[zid] = (dx, dy)
        info = rot_info[idx] if idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
        zone_rot[zid] = float(info.get("angle", 0.0))
        zone_center[zid] = (float(info.get("cx", 0.0)), float(info.get("cy", 0.0)))
        # muted: noisy progress logs during reload

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="none" stroke="none"/>')
    parts.append('<g id="root">')
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
        # muted: noisy progress logs during reload
    parts.append("</g>")
    parts.append('<g id="bleed">')
    if include_bleed and bleed_canvas > 0:
        # Projection-grid pipeline (same family as zone_0_projection_grid.svg):
        # region vertices near zone boundary -> projection -> weld/merge in boundary order -> bleed polygons.
        projection_threshold = float(bleed_canvas)
        weld_threshold = 1.0
        merge_threshold = 0.5
        zone_regions: Dict[int, List[int]] = {}
        zone_poly_by_zid: Dict[int, List[Tuple[float, float]]] = {}
        for rid, zid in enumerate(zone_id):
            zone_regions.setdefault(int(zid), []).append(rid)
        for idx, zid in enumerate(zone_order):
            if idx < len(zone_polys):
                zone_poly_by_zid[int(zid)] = list(zone_polys[idx])

        for zid, rid_list in zone_regions.items():
            if zid not in zone_shift:
                continue
            src_zone = zone_poly_by_zid.get(zid) or []
            if len(src_zone) < 3:
                continue
            zone_base = src_zone[:-1] if len(src_zone) > 1 and src_zone[0] == src_zone[-1] else src_zone
            if len(zone_base) < 3:
                continue

            dx, dy = zone_shift[zid]
            ang = zone_rot.get(zid, 0.0)
            cx, cy = zone_center.get(zid, (0.0, 0.0))
            boundary = [(p[0] + dx, p[1] + dy) for p in _rotate_pts(zone_base, ang, cx, cy)]
            if len(boundary) < 3:
                continue
            boundary_off = _offset_outline_same_vertices(boundary, bleed_canvas)
            if len(boundary_off) != len(boundary):
                continue

            n = len(boundary)
            edge_len: List[float] = []
            cum = [0.0]
            for i in range(n):
                a = boundary[i]
                b = boundary[(i + 1) % n]
                ll = float(np.hypot(b[0] - a[0], b[1] - a[1]))
                edge_len.append(ll)
                cum.append(cum[-1] + ll)
            perimeter = cum[-1]
            if perimeter <= 1e-6:
                continue

            def _pt_at(edge_idx: int, t: float, seq: List[Tuple[float, float]]) -> Tuple[float, float]:
                i0 = max(0, min(n - 1, int(edge_idx)))
                i1 = (i0 + 1) % n
                tt = max(0.0, min(1.0, float(t)))
                p0 = seq[i0]
                p1 = seq[i1]
                return (p0[0] + (p1[0] - p0[0]) * tt, p0[1] + (p1[1] - p0[1]) * tt)

            def _project_param(pt: Tuple[float, float]):
                best = None
                best_d2 = None
                for i in range(n):
                    a = boundary[i]
                    b = boundary[(i + 1) % n]
                    vx = b[0] - a[0]
                    vy = b[1] - a[1]
                    ll2 = vx * vx + vy * vy
                    if ll2 <= 1e-12:
                        continue
                    t = ((pt[0] - a[0]) * vx + (pt[1] - a[1]) * vy) / ll2
                    t = max(0.0, min(1.0, t))
                    px = a[0] + vx * t
                    py = a[1] + vy * t
                    ddx = px - pt[0]
                    ddy = py - pt[1]
                    d2 = ddx * ddx + ddy * ddy
                    if best_d2 is None or d2 < best_d2:
                        s = cum[i] + edge_len[i] * t
                        best = {"x": px, "y": py, "edge": i, "t": t, "s": s, "d2": d2}
                        best_d2 = d2
                return best

            def _param_from_s(s: float):
                ss = float(s) % perimeter
                i = 0
                while i + 1 < len(cum) and cum[i + 1] < ss - 1e-9:
                    i += 1
                i = min(i, n - 1)
                ll = edge_len[i]
                t = 0.0 if ll <= 1e-9 else (ss - cum[i]) / ll
                return i, max(0.0, min(1.0, t)), ss

            region_items: List[Tuple[int, Polygon, List[Tuple[float, float]]]] = []
            for rid in rid_list:
                if rid >= len(polys):
                    continue
                src = polys[rid]
                if len(src) < 3:
                    continue
                moved = [(p[0] + dx, p[1] + dy) for p in _rotate_pts(src, ang, cx, cy)]
                try:
                    rp = Polygon(moved)
                    if rp.is_empty:
                        continue
                    if not rp.is_valid:
                        rp = make_valid(rp)
                    if rp.is_empty or not isinstance(rp, Polygon):
                        continue
                except Exception:
                    continue
                region_items.append((rid, rp, moved))
            if not region_items:
                continue

            vertices = [{"kind": "v", "edge": i, "t": 0.0, "s": cum[i], "x": boundary[i][0], "y": boundary[i][1]} for i in range(n)]
            projections = []
            for rid, _rp, moved in region_items:
                for p in moved:
                    pr = _project_param(p)
                    if pr is None:
                        continue
                    if pr["d2"] > projection_threshold * projection_threshold + 1e-9:
                        continue
                    pr["kind"] = "p"
                    pr["rid"] = rid
                    projections.append(pr)
            if not projections:
                continue

            # Weld projection points to zone vertices first.
            for pr in projections:
                snap_vi = None
                snap_d2 = weld_threshold * weld_threshold
                for vi in range(n):
                    vx, vy = boundary[vi]
                    ddx = pr["x"] - vx
                    ddy = pr["y"] - vy
                    d2 = ddx * ddx + ddy * ddy
                    if d2 <= snap_d2:
                        snap_d2 = d2
                        snap_vi = vi
                if snap_vi is not None:
                    pr["edge"] = snap_vi
                    pr["t"] = 0.0
                    pr["s"] = cum[snap_vi]
                    pr["x"] = boundary[snap_vi][0]
                    pr["y"] = boundary[snap_vi][1]

            projections = sorted(projections, key=lambda e: e["s"])
            welded = []
            for pr in projections:
                if not welded:
                    welded.append(dict(pr))
                    continue
                last = welded[-1]
                ds = abs(float(pr["s"]) - float(last["s"]))
                dxy = float(np.hypot(float(pr["x"]) - float(last["x"]), float(pr["y"]) - float(last["y"])))
                if ds <= weld_threshold or dxy <= weld_threshold:
                    s_new = 0.5 * (float(pr["s"]) + float(last["s"]))
                    ei, tt, ss = _param_from_s(s_new)
                    px, py = _pt_at(ei, tt, boundary)
                    last["edge"] = ei
                    last["t"] = tt
                    last["s"] = ss
                    last["x"] = px
                    last["y"] = py
                else:
                    welded.append(dict(pr))

            ordered = vertices + welded
            ordered = sorted(ordered, key=lambda e: e["s"])
            merged = []
            for item in ordered:
                if not merged:
                    merged.append(dict(item))
                    continue
                prev = merged[-1]
                ds = abs(float(item["s"]) - float(prev["s"]))
                dxy = float(np.hypot(float(item["x"]) - float(prev["x"]), float(item["y"]) - float(prev["y"])))
                if ds <= merge_threshold or dxy <= merge_threshold:
                    if prev.get("kind") != "v" and item.get("kind") == "v":
                        merged[-1] = dict(item)
                else:
                    merged.append(dict(item))
            if len(merged) < 2:
                continue

            for i in range(len(merged)):
                a = merged[i]
                b = merged[(i + 1) % len(merged)]
                sa = float(a["s"])
                sb = float(b["s"])
                ds = sb - sa
                if ds <= 0:
                    ds += perimeter
                if ds <= 1e-6:
                    continue
                ax, ay = float(a["x"]), float(a["y"])
                bx, by = float(b["x"]), float(b["y"])
                aox, aoy = _pt_at(int(a["edge"]), float(a["t"]), boundary_off)
                box, boy = _pt_at(int(b["edge"]), float(b["t"]), boundary_off)
                bleed_poly = [(ax, ay), (aox, aoy), (box, boy), (bx, by)]
                try:
                    pg = Polygon(bleed_poly)
                except Exception:
                    continue
                if pg.is_empty or float(pg.area) <= 1e-8:
                    continue

                mx = 0.5 * (ax + bx)
                my = 0.5 * (ay + by)
                mid = Point(mx, my)
                best_rid = None
                best_d = None
                for rid, rp, _ in region_items:
                    try:
                        d = float(rp.boundary.distance(mid))
                    except Exception:
                        continue
                    if best_d is None or d < best_d:
                        best_d = d
                        best_rid = rid
                bgr = colors[best_rid] if best_rid is not None and best_rid < len(colors) else (200, 200, 200)
                bb, bg, br = bgr
                d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in bleed_poly) + " Z"
                parts.append(f'<path d="{d}" fill="rgb({br},{bg},{bb})" stroke="none"/>')
    parts.append("</g>")
    parts.append("</g></svg>")
    out_svg = out_path if out_path is not None else config.OUT_PACK_SVG
    out_svg.write_text("".join(parts), encoding="utf-8")


def write_empty_pack_svg(canvas: Tuple[int, int], out_path: Path) -> None:
    w, h = canvas
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<g id="root">',
        '<g id="fill"></g>',
        '<g id="bleed"></g>',
        "</g></svg>",
    ]
    out_path.write_text("".join(parts), encoding="utf-8")


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
    # Raster output disabled (cv2 removed).
    return


def compute_scene(svg_path, snap: float, render_packed_png: bool = False) -> Dict:
    config._apply_pack_env()
    pack_mode = str(getattr(config, "PACK_MODE", "fast") or "fast").strip().lower()
    angle_step = max(1.0, float(getattr(config, "PACK_ANGLE_STEP", 5.0) or 5.0))
    grid_step = max(1.0, float(getattr(config, "PACK_GRID_STEP", 5.0) or 5.0))
    regions, polys, canvas, debug = geometry.build_regions_from_svg(svg_path, snap_override=snap)
    zone_id = zones.build_zones(polys, config.TARGET_ZONES)
    zone_id, zone_members = zones._remap_zones_by_area(polys, zone_id)
    zone_boundaries = zones.build_zone_boundaries(polys, zone_id)
    zone_geoms = zones.build_zone_geoms(polys, zone_id)
    zone_polys, zone_order, zone_poly_debug = zones.build_zone_polys(polys, zone_id)
    zone_pack_polys = _build_zone_pack_polys(zone_polys, float(config.PACK_BLEED), bevel_angle=60.0)
    zone_pack_angles = [_min_bbox_align_angle(p) for p in zone_pack_polys]
    zone_pack_centers = [
        (float(Polygon(p).centroid.x), float(Polygon(p).centroid.y)) if p else (0.0, 0.0)
        for p in zone_pack_polys
    ]
    zone_ids = sorted(zone_geoms.keys())
    rng = np.random.default_rng(42)
    shuffled = zone_ids.copy()
    rng.shuffle(shuffled)
    zone_label_map = {z: idx + 1 for idx, z in enumerate(shuffled)}
    mode = pack_mode if pack_mode in {"fast", "balanced", "tight", "nested"} else "balanced"
    # Nested brute-force is expensive on large zone counts; keep UI responsive by
    # using fast rectpack + polygon compaction for fast/balanced and auto-fallback.
    use_true_nested = mode in {"tight", "nested"} and len(zone_pack_polys) <= 70
    quick_one_page_done = False
    if mode == "fast" and zone_pack_polys:
        placements, order, rot_info = pack_regions_raster_fast(
            zone_pack_polys,
            canvas,
            fixed_centers=zone_pack_centers,
            grid_step=max(3.0, min(6.0, float(grid_step))),
            search_stride=1,
            safety_padding=max(
                3.0,
                float(getattr(config, "PACK_BLEED", 3.0)) * 0.9,
                float(getattr(config, "PADDING", 0.0)) + 1.0,
            ),
        )
        # Raster fit can produce tiny vector overlaps; run a lightweight geometric
        # separation pass to enforce visible gaps.
        fast_gap = max(
            3.0,
            float(getattr(config, "PACK_BLEED", 3.0)) * 0.9,
            float(getattr(config, "PADDING", 0.0)) + 1.0,
        )
        placements = _resolve_pack_overlaps(
            zone_pack_polys,
            placements,
            rot_info,
            step=1.0,
            padding=fast_gap,
            max_iter=60,
        )
        quick_one_page_done = True
    if not use_true_nested and zone_pack_polys and not quick_one_page_done:
        # Stage 1 (fast): try to cram everything into page 1 using coarse search first.
        # If this already places all zones, skip expensive refinement passes.
        quick_angle = max(angle_step, 30.0 if mode == "fast" else 15.0)
        quick_grid = max(grid_step, 12.0 if mode == "fast" else 8.0)
        q_placements, q_order, q_rot_info = pack_regions(
            zone_pack_polys,
            canvas,
            allow_rotate=True,
            angle_step=quick_angle,
            grid_step=quick_grid,
            fixed_angles=None,
            fixed_centers=zone_pack_centers,
            max_bins=1,
            try_heuristics=True,
            two_pass=False,
            preferred_indices=None,
            use_gap_only=False,
        )
        q_placed = sum(1 for dx, dy, bw, bh, _ in q_placements if dx >= 0 and dy >= 0 and bw > 0 and bh > 0)
        if q_placed >= len(zone_pack_polys):
            placements, order, rot_info = q_placements, q_order, q_rot_info
            quick_one_page_done = True
    if use_true_nested:
        placements, order, rot_info = pack_regions_nested(
            zone_pack_polys,
            canvas,
            angle_step=angle_step,
            grid_step=grid_step,
            fixed_centers=zone_pack_centers,
            max_bins=2,
        )
    elif not quick_one_page_done:
        placements, order, rot_info = pack_regions(
            zone_pack_polys,
            canvas,
            allow_rotate=True,
            angle_step=angle_step,
            grid_step=grid_step,
            fixed_angles=None,
            fixed_centers=zone_pack_centers,
            # Force page-1-first packing; spill to page2 only for true remainder.
            max_bins=1,
            try_heuristics=True,
            two_pass=True,
            preferred_indices=None,
            use_gap_only=False,
        )
        if mode == "fast":
            nest_step = max(1.0, grid_step)
            nest_passes = 1
        elif mode == "tight":
            nest_step = max(0.5, grid_step * 0.5)
            nest_passes = 3
        else:
            nest_step = max(0.75, grid_step * 0.75)
            nest_passes = 2
        placements = compact_nesting_polygons(
            zone_pack_polys,
            placements,
            rot_info,
            canvas,
            step=nest_step,
            passes=nest_passes,
        )
        # Final rescue pass: try to move page2 leftovers into page1 true polygon gaps.
        placements, rot_info, moved_back = repack_page2_into_page1_nested(
            zone_pack_polys,
            placements,
            rot_info,
            canvas,
            angle_step=angle_step,
            grid_step=max(1.0, nest_step),
        )
        if moved_back > 0:
            order = [i for i, p in enumerate(placements) if p[2] > 0 and p[3] > 0]
    # Final anti-collision pass, aligned with TS raster-mask intent:
    # enforce a strict geometric clearance before rendering.
    strict_gap = max(1.0, float(getattr(config, "PADDING", 0.0)))
    placements = _resolve_pack_overlaps(
        zone_pack_polys,
        placements,
        rot_info,
        step=1.0,
        padding=strict_gap,
        max_iter=120,
    )
    placements = compact_nesting_polygons(
        zone_pack_polys,
        placements,
        rot_info,
        canvas,
        step=0.75,
        passes=1,
    )
    placements = _resolve_pack_overlaps(
        zone_pack_polys,
        placements,
        rot_info,
        step=0.75,
        padding=strict_gap,
        max_iter=40,
    )
    placements = _fit_placements_into_canvas(placements, rot_info, canvas)
    zone_labels = {}
    zone_index = {z: idx for idx, z in enumerate(zone_order)}
    for zid, geom in zone_geoms.items():
        lx = None
        ly = None
        idx = zone_index.get(zid, None)
        if idx is not None and idx < len(zone_polys):
            border_pts = zone_polys[idx]
            if border_pts:
                try:
                    border_poly = Polygon(border_pts)
                    c = border_poly.centroid
                    cx = float(c.x)
                    cy = float(c.y)
                    half_w = 1.2
                    half_h = 1.2
                    test_pts = [
                        (cx, cy),
                        (cx - half_w, cy - half_h),
                        (cx + half_w, cy - half_h),
                        (cx - half_w, cy + half_h),
                        (cx + half_w, cy + half_h),
                        (cx, cy - half_h),
                        (cx, cy + half_h),
                        (cx - half_w, cy),
                        (cx + half_w, cy),
                    ]
                    if all(geom.covers(Point(px, py)) for px, py in test_pts):
                        lx, ly = cx, cy
                except Exception:
                    pass
        if lx is None or ly is None:
            members = zone_members.get(zid, [])
            if members:
                rid0 = members[0]
                c = Polygon(polys[rid0]).centroid
                lx, ly = float(c.x), float(c.y)
            else:
                lx, ly = zones._label_pos_for_zone(geom)
        zone_labels[str(zid)] = {"x": lx, "y": ly, "label": zone_label_map.get(zid, zid)}

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

    zone_shift: Dict[int, Tuple[float, float]] = {}
    zone_rot: Dict[int, float] = {}
    zone_center: Dict[int, Tuple[float, float]] = {}
    placement_bin = [int(info.get("bin", -1)) for info in rot_info]
    placement_bin_by_zid: Dict[int, int] = {}
    for idx, zid in enumerate(zone_order):
        if idx >= len(placements):
            continue
        dx, dy, _, _, _ = placements[idx]
        zone_shift[zid] = (dx, dy)
        info = rot_info[idx] if idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
        zone_rot[zid] = float(info.get("angle", 0.0))
        zone_center[zid] = (float(info.get("cx", 0.0)), float(info.get("cy", 0.0)))
        if idx < len(placement_bin):
            placement_bin_by_zid[zid] = placement_bin[idx]

    # Center all placed zones to canvas center based on current bounds.
    try:
        minx = None
        miny = None
        maxx = None
        maxy = None
        for idx, zid in enumerate(zone_order):
            if idx >= len(placements) or idx >= len(rot_info):
                continue
            dx, dy, bw, bh, _ = placements[idx]
            if bw <= 0 or bh <= 0:
                continue
            info = rot_info[idx]
            x0 = float(info.get("minx", 0.0)) + dx
            y0 = float(info.get("miny", 0.0)) + dy
            x1 = x0 + bw
            y1 = y0 + bh
            minx = x0 if minx is None else min(minx, x0)
            miny = y0 if miny is None else min(miny, y0)
            maxx = x1 if maxx is None else max(maxx, x1)
            maxy = y1 if maxy is None else max(maxy, y1)
        if minx is not None and miny is not None and maxx is not None and maxy is not None:
            cx = (minx + maxx) / 2.0
            cy = (miny + maxy) / 2.0
            dxc = (canvas[0] / 2.0) - cx
            dyc = (canvas[1] / 2.0) - cy
            if abs(dxc) > 1e-6 or abs(dyc) > 1e-6:
                for idx, zid in enumerate(zone_order):
                    if idx >= len(placements):
                        continue
                    dx, dy, bw, bh, rot = placements[idx]
                    if bw <= 0 or bh <= 0:
                        continue
                    placements[idx] = (int(dx + dxc), int(dy + dyc), bw, bh, rot)
                    if zid in zone_shift:
                        zone_shift[zid] = (zone_shift[zid][0] + dxc, zone_shift[zid][1] + dyc)
    except Exception:
        pass

    # No 180 rotation for packed output.

    colors, _ = geometry.compute_region_colors(polys, canvas)
    region_colors = [f"#{r:02x}{g:02x}{b:02x}" for (b, g, r) in colors]

    missing = [z for z in zone_order if z not in zone_shift]
    zone_index = {zid: idx for idx, zid in enumerate(zone_order)}
    lines = [
        f"zones_total={len(zone_order)}",
        f"placed={len(zone_shift)}",
        f"missing={len(missing)}",
    ]
    for idx, zid in enumerate(zone_order):
        label = zone_label_map.get(zid, zid)
        if zid in zone_shift:
            dx, dy = zone_shift[zid]
            ang = zone_rot.get(zid, 0.0)
            cx, cy = zone_center.get(zid, (0.0, 0.0))
            lines.append(
                f"zone_id={zid} shuffle_label={label} dx={dx:.2f} dy={dy:.2f} angle={ang:.2f} cx={cx:.2f} cy={cy:.2f}"
            )
        else:
            if idx < len(placements):
                dx, dy, _, _, _ = placements[idx]
            else:
                dx = dy = -1
            info = rot_info[idx] if idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
            ang = float(info.get("angle", 0.0))
            cx = float(info.get("cx", 0.0))
            cy = float(info.get("cy", 0.0))
            lines.append(
                f"zone_id={zid} shuffle_label={label} missing=1 dx={dx:.2f} dy={dy:.2f} angle={ang:.2f} cx={cx:.2f} cy={cy:.2f}"
            )
    if render_packed_png:
        config.OUT_PACK_RASTER_LOG.write_text("\n".join(lines), encoding="utf-8")

    missing = [z for z in zone_order if z not in zone_shift]
    lines = [
        f"zones_total={len(zone_order)}",
        f"placed={len(zone_shift)}",
        f"missing={len(missing)}",
    ]
    for zid in missing:
        idx = zone_index.get(zid, -1)
        if 0 <= idx < len(placements):
            dx, dy, _, _, _ = placements[idx]
        else:
            dx = dy = -1
        info = rot_info[idx] if 0 <= idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
        ang = float(info.get("angle", 0.0))
        cx = float(info.get("cx", 0.0))
        cy = float(info.get("cy", 0.0))
        lines.append(
            f"zone_id={zid} shuffle_label={zone_label_map.get(zid, zid)} dx={dx:.2f} dy={dy:.2f} angle={ang:.2f} cx={cx:.2f} cy={cy:.2f}"
        )
    config.OUT_PACK_MISSING_LOG.write_text("\n".join(lines), encoding="utf-8")

    write_pack_log(zone_pack_polys, placements, rot_info, zone_label_map, config.OUT_PACK_LOG, canvas)

    placement_bin = [int(info.get("bin", -1)) for info in rot_info]
    order_page0 = [i for i, b in enumerate(placement_bin) if b == 0]
    order_page1 = [i for i, b in enumerate(placement_bin) if b == 1]
    placement_bin_by_zid = {
        zid: placement_bin[idx] for idx, zid in enumerate(zone_order) if idx < len(placement_bin)
    }
    placement_bin_by_zid = {zid: placement_bin[idx] for idx, zid in enumerate(zone_order) if idx < len(placement_bin)}

    write_pack_bbox_svg(
        placements,
        rot_info,
        canvas,
        config.OUT_PACK_BBOX_SVG,
        packed_order=order_page0,
    )
    write_pack_bbox_vs_poly_svg(
        zone_pack_polys,
        placements,
        rot_info,
        canvas,
        config.OUT_PACK_BBOX_VS_POLY_SVG,
        zone_label_map=zone_label_map,
        packed_order=order_page0,
    )
    if order_page1:
        write_pack_bbox_svg(
            placements,
            rot_info,
            canvas,
            config.OUT_PACK_BBOX_SVG_PAGE2,
            packed_order=order_page1,
        )
        write_pack_bbox_vs_poly_svg(
            zone_pack_polys,
            placements,
            rot_info,
            canvas,
            config.OUT_PACK_BBOX_VS_POLY_SVG_PAGE2,
            zone_label_map=zone_label_map,
            packed_order=order_page1,
        )
    else:
        write_pack_bbox_svg(placements, rot_info, canvas, config.OUT_PACK_BBOX_SVG_PAGE2, packed_order=[])
        write_pack_bbox_vs_poly_svg(
            zone_pack_polys,
            placements,
            rot_info,
            canvas,
            config.OUT_PACK_BBOX_VS_POLY_SVG_PAGE2,
            zone_label_map=zone_label_map,
            packed_order=[],
        )

    write_pack_svg(
        polys,
        zone_id,
        zone_order,
        zone_polys,
        placements,
        canvas,
        colors,
        rot_info,
        placement_bin=placement_bin,
        placement_bin_by_zid=placement_bin_by_zid,
        page_idx=0,
        out_path=config.OUT_PACK_SVG,
    )
    if order_page1:
        write_pack_svg(
            polys,
            zone_id,
            zone_order,
            zone_polys,
            placements,
            canvas,
            colors,
            rot_info,
            placement_bin=placement_bin,
            placement_bin_by_zid=placement_bin_by_zid,
            page_idx=1,
            out_path=config.OUT_PACK_SVG_PAGE2,
        )
    else:
        write_empty_pack_svg(canvas, config.OUT_PACK_SVG_PAGE2)

    if render_packed_png:
        write_pack_png(
            polys,
            zone_id,
            zone_order,
            zone_polys,
            placements,
            canvas,
            colors,
            zone_geoms,
            zone_label_map,
            region_labels,
            rot_info,
        )

    debug["zones_total"] = float(max(zone_id) + 1) if zone_id else 0.0
    debug["packed_placed"] = float(len(zone_shift))
    debug["zones_empty"] = zone_poly_debug.get("empty", [])
    debug["zones_convex_hull"] = zone_poly_debug.get("convex_hull", [])

    return {
        "canvas": {"w": canvas[0], "h": canvas[1]},
        "draw_scale": config.DRAW_SCALE,
        "regions": polys,
        "zone_boundaries": zone_boundaries,
        "zone_id": zone_id,
        "zone_labels": zone_labels,
        "region_labels": region_labels,
        "zone_order": zone_order,
        "zone_pack_polys": zone_pack_polys,
        "zone_pack_angles": zone_pack_angles,
        "zone_rot": zone_rot,
        "zone_center": zone_center,
        "zone_shift": zone_shift,
        "zone_label_map": zone_label_map,
        "placement_bin": placement_bin_by_zid,
        "region_colors": region_colors,
        "colors_bgr": [[int(b), int(g), int(r)] for (b, g, r) in colors],
        "placements": [[int(a), int(b), int(c), int(d), bool(e)] for (a, b, c, d, e) in placements],
        "rot_info": [
            {
                "angle": float(info.get("angle", 0.0)),
                "cx": float(info.get("cx", 0.0)),
                "cy": float(info.get("cy", 0.0)),
            }
            for info in rot_info
        ],
        "debug": debug,
        "snap": snap,
        "pack_mode": pack_mode,
        "pack_grid": grid_step,
        "pack_angle": angle_step,
    }


def main() -> None:
    if not config.SVG_PATH.exists():
        raise SystemExit(f"Missing {config.SVG_PATH}")
    config._apply_pack_env()

    svg_mtime = os.path.getmtime(config.SVG_PATH)
    cache_ok = False
    if config.USE_ZONE_CACHE and config.OUT_ZONES_JSON.exists():
        try:
            data = json.loads(config.OUT_ZONES_JSON.read_text(encoding="utf-8"))
            cache_ok = float(data.get("svg_mtime", -1)) >= svg_mtime
        except Exception:
            cache_ok = False

    if cache_ok:
        polys, zone_id = zones.load_zones_cache(config.OUT_ZONES_JSON)
        base_canvas = svg_utils._get_canvas_size(ET.parse(config.SVG_PATH).getroot(), 1.0)
        canvas = base_canvas
        regions = [geometry.RegionInfo(i, 0.0, (0, 0, 0, 0), (0.0, 0.0)) for i in range(len(polys))]
    else:
        regions, polys, canvas, _ = geometry.build_regions_from_svg(config.SVG_PATH)
        geometry.write_log(regions, config.OUT_LOG)
        geometry.write_png(polys, regions, canvas)

        zone_id = zones.build_zones(polys, config.TARGET_ZONES)
        zone_id, _ = zones._remap_zones_by_area(polys, zone_id)
        zones.write_zones_log(zone_id, config.OUT_ZONES_LOG)
        zones.save_zones_cache(zone_id, polys, config.OUT_ZONES_JSON)

    colors, _ = geometry.render_color_regions(polys, svg_utils._get_canvas_size(ET.parse(config.SVG_PATH).getroot(), 1.0))
    geometry.write_zones_png(polys, zone_id, canvas, colors)

    zone_polys, zone_order, _ = zones.build_zone_polys(polys, zone_id)
    zone_geoms = zones.build_zone_geoms(polys, zone_id)
    zone_ids = sorted(zone_geoms.keys())
    rng = np.random.default_rng(42)
    shuffled = zone_ids.copy()
    rng.shuffle(shuffled)
    zone_labels = {z: idx + 1 for idx, z in enumerate(shuffled)}
    zone_boundaries = zones.build_zone_boundaries(polys, zone_id)
    zones.write_zone_outline_png(zone_geoms, zone_labels, canvas, zone_boundaries)
    svg_utils.write_zone_svg(polys, zone_boundaries, canvas, colors)
    svg_utils.write_zone_outline_svg(zone_boundaries, canvas)
    svg_utils.write_region_svg(polys, canvas)
    zones.write_zones_log(zone_id, config.OUT_ZONES_LOG, zone_labels)

    region_ids = list(range(len(polys)))
    rng = np.random.default_rng(43)
    rng.shuffle(region_ids)
    region_labels = {rid: idx + 1 for idx, rid in enumerate(region_ids)}
    base_canvas = svg_utils._get_canvas_size(ET.parse(config.SVG_PATH).getroot(), 1.0)
    zone_pack_polys = _build_zone_pack_polys(zone_polys, float(config.PACK_BLEED), bevel_angle=60.0)
    zone_pack_angles = [_min_bbox_align_angle(p) for p in zone_pack_polys]
    zone_pack_centers = [
        (float(Polygon(p).centroid.x), float(Polygon(p).centroid.y)) if p else (0.0, 0.0)
        for p in zone_pack_polys
    ]
    placements, order, rot_info = pack_regions(
        zone_pack_polys,
        base_canvas,
        allow_rotate=True,
        angle_step=5.0,
        fixed_angles=None,
        fixed_centers=zone_pack_centers,
        # Keep CLI/debug path consistent with page-1-first behavior.
        max_bins=1,
        try_heuristics=True,
        two_pass=True,
        preferred_indices=None,
        use_gap_only=False,
    )
    write_pack_log(zone_pack_polys, placements, rot_info, zone_labels, config.OUT_PACK_LOG, base_canvas)
    write_pack_log(zone_pack_polys, placements, rot_info, zone_labels, config.OUT_PACK_LOG, base_canvas)
    placement_bin = [int(info.get("bin", -1)) for info in rot_info]
    placement_bin_by_zid = {
        zid: placement_bin[idx]
        for idx, zid in enumerate(zone_order)
        if idx < len(placement_bin)
    }
    order_page0 = [i for i, b in enumerate(placement_bin) if b == 0]
    order_page1 = [i for i, b in enumerate(placement_bin) if b == 1]
    write_pack_bbox_svg(
        placements,
        rot_info,
        base_canvas,
        config.OUT_PACK_BBOX_SVG,
        packed_order=order_page0,
    )
    write_pack_bbox_vs_poly_svg(
        zone_pack_polys,
        placements,
        rot_info,
        base_canvas,
        config.OUT_PACK_BBOX_VS_POLY_SVG,
        zone_label_map=zone_labels,
        packed_order=order_page0,
    )
    if order_page1:
        write_pack_bbox_svg(
            placements,
            rot_info,
            base_canvas,
            config.OUT_PACK_BBOX_SVG_PAGE2,
            packed_order=order_page1,
        )
        write_pack_bbox_vs_poly_svg(
            zone_pack_polys,
            placements,
            rot_info,
            base_canvas,
            config.OUT_PACK_BBOX_VS_POLY_SVG_PAGE2,
            zone_label_map=zone_labels,
            packed_order=order_page1,
        )
    else:
        write_pack_bbox_svg(placements, rot_info, base_canvas, config.OUT_PACK_BBOX_SVG_PAGE2, packed_order=[])
        write_pack_bbox_vs_poly_svg(
            zone_pack_polys,
            placements,
            rot_info,
            base_canvas,
            config.OUT_PACK_BBOX_VS_POLY_SVG_PAGE2,
            zone_label_map=zone_labels,
            packed_order=[],
        )
    write_pack_svg(
        polys,
        zone_id,
        zone_order,
        zone_polys,
        placements,
        base_canvas,
        colors,
        rot_info,
        placement_bin=placement_bin,
        placement_bin_by_zid=placement_bin_by_zid,
        page_idx=0,
        out_path=config.OUT_PACK_SVG,
    )
    if order_page1:
        write_pack_svg(
            polys,
            zone_id,
            zone_order,
            zone_polys,
            placements,
            base_canvas,
            colors,
            rot_info,
            placement_bin=placement_bin,
            placement_bin_by_zid=placement_bin_by_zid,
            page_idx=1,
            out_path=config.OUT_PACK_SVG_PAGE2,
        )
    else:
        write_empty_pack_svg(base_canvas, config.OUT_PACK_SVG_PAGE2)

    total_zones = max(zone_id) + 1 if zone_id else 0
    print(
        f"Wrote {config.OUT_LOG}, {config.OUT_PNG}, {config.OUT_ZONES_LOG}, {config.OUT_ZONES_PNG}, {config.OUT_PACK_LOG} "
        f"with {len(regions)} regions and {total_zones} zones"
    )


if __name__ == "__main__":
    main()
