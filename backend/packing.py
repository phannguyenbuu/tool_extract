from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from math import ceil
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
from shapely.affinity import rotate as _srotate, translate as _stranslate
from shapely.geometry import Polygon, Point
from shapely.geometry.base import BaseGeometry
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
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += (x1 * y2) - (x2 * y1)
    ccw = area > 0

    lines = []
    for i in range(len(pts)):
        p0, p1 = pts[i], pts[(i + 1) % len(pts)]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        ln = float(np.hypot(dx, dy))
        if ln <= 1e-6:
            lines.append(None); continue
        nx, ny = (dy/ln, -dx/ln) if ccw else (-dy/ln, dx/ln)
        lines.append(((p0[0]+nx*offset, p0[1]+ny*offset), (p1[0]+nx*offset, p1[1]+ny*offset), (nx, ny)))

    out = []
    for i in range(len(pts)):
        prev, cur = lines[(i - 1) % len(pts)], lines[i]
        if prev is None or cur is None: out.append(pts[i]); continue
        (p1, p2, n1), (p3, p4, n2) = prev, cur
        
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-8:
            out.append((pts[i][0] + n2[0] * offset, pts[i][1] + n2[1] * offset)); continue
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/den
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/den
        out.append((px, py))
    return out


def _bevel_corner(prev_pt, cur_pt, next_pt, r) -> List[Tuple[float, float]]:
    if r <= 0: return [cur_pt]
    v1x, v1y = prev_pt[0] - cur_pt[0], prev_pt[1] - cur_pt[1]
    v2x, v2y = next_pt[0] - cur_pt[0], next_pt[1] - cur_pt[1]
    l1, l2 = float(np.hypot(v1x, v1y)), float(np.hypot(v2x, v2y))
    if l1 <= 1e-6 or l2 <= 1e-6: return [cur_pt]
    d1, d2 = min(r, l1 * 0.49), min(r, l2 * 0.49)
    return [(cur_pt[0] + v1x/l1*d1, cur_pt[1] + v1y/l1*d1), (cur_pt[0] + v2x/l2*d2, cur_pt[1] + v2y/l2*d2)]


def _bevel_outline_by_angle(polyline, r, angle_thresh=90.0) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    if r <= 0 or len(polyline) < 3: return polyline, []
    closed = abs(polyline[0][0] - polyline[-1][0]) < 1e-6 and abs(polyline[0][1] - polyline[-1][1]) < 1e-6
    pts = polyline[:-1] if closed else polyline
    n, out, debug_pts = len(pts), [], []
    if n < 3: return polyline, []
    for i in range(n):
        p_prev, p_cur, p_next = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        v1x, v1y = p_prev[0]-p_cur[0], p_prev[1]-p_cur[1]
        v2x, v2y = p_next[0]-p_cur[0], p_next[1]-p_cur[1]
        l1, l2 = float(np.hypot(v1x, v1y)), float(np.hypot(v2x, v2y))
        if l1 > 1e-6 and l2 > 1e-6:
            dot = max(-1.0, min(1.0, (v1x*v2x + v1y*v2y) / (l1*l2)))
            if float(np.degrees(np.arccos(dot))) < angle_thresh:
                scale = (angle_thresh / max(float(np.degrees(np.arccos(dot))), 1e-3)) ** 1.5
                out.extend(_bevel_corner(p_prev, p_cur, p_next, min(r * scale, r * 6.0)))
                debug_pts.append(p_cur); continue
        out.append(p_cur)
    if closed: out.append(out[0])
    return out, debug_pts


def _build_zone_pack_polys(zone_polys, bleed) -> List[List[Tuple[float, float]]]:
    if not zone_polys: return []
    out = []
    for poly in zone_polys:
        pts = poly[:-1] if len(poly)>1 and abs(poly[0][0]-poly[-1][0])<1e-6 and abs(poly[0][1]-poly[-1][1])<1e-6 else poly[:]
        if len(pts) >= 3:
            if bleed > 0: pts = _offset_outline_same_vertices(pts, bleed)
            out.append(pts)
    return out

def _resolve_pack_overlaps(zone_polys, placements, rot_info, step, padding=0.0, max_iter=50):
    if not zone_polys or not placements: return placements
    n, out = min(len(zone_polys), len(placements)), list(placements)
    for _ in range(max_iter):
        moved, tpolys, centroids = False, [], []
        for i in range(n):
            dx, dy = out[i][0], out[i][1]
            info = rot_info[i] if i < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
            rpts = _rotate_pts(zone_polys[i], info.get("angle", 0.0), info.get("cx", 0.0), info.get("cy", 0.0))
            poly = Polygon([(p[0] + dx, p[1] + dy) for p in rpts])
            if padding > 0: poly = poly.buffer(padding)
            tpolys.append(poly)
            centroids.append((poly.centroid.x, poly.centroid.y) if not poly.is_empty else (0,0))
        for i in range(n):
            if tpolys[i].is_empty: continue
            for j in range(i + 1, n):
                if tpolys[j].is_empty or not tpolys[i].intersects(tpolys[j]): continue
                moved = True
                vx, vy = centroids[j][0]-centroids[i][0], centroids[j][1]-centroids[i][1]
                ln = float(np.hypot(vx, vy))
                ux, uy = (vx/ln, vy/ln) if ln>1e-6 else (1.0, 0.0)
                dxj, dyj, wj, hj, rfj = out[j]
                out[j] = (dxj + ux * step, dyj + uy * step, wj, hj, rfj)
        if not moved: break
    return out


def _fit_placements_into_canvas(placements, rot_info, canvas):
    if not placements or not rot_info: return placements
    w, h, out = float(canvas[0]), float(canvas[1]), list(placements)
    bins = {}
    for i in range(min(len(out), len(rot_info))):
        if out[i][2] > 0: bins.setdefault(int(rot_info[i].get("bin", 0)), []).append(i)
    for idxs in bins.values():
        minx = miny = maxx = maxy = None
        for i in idxs:
            x0, y0 = float(rot_info[i].get("minx", 0.0)) + out[i][0], float(rot_info[i].get("miny", 0.0)) + out[i][1]
            minx = x0 if minx is None else min(minx, x0)
            miny = y0 if miny is None else min(miny, y0)
            maxx = x0 + out[i][2] if maxx is None else max(maxx, x0 + out[i][2])
            maxy = y0 + out[i][3] if maxy is None else max(maxy, y0 + out[i][3])
        if minx is None: continue
        tx, ty = max(0.0, -minx) + min(0.0, w - maxx), max(0.0, -miny) + min(0.0, h - maxy)
        for i in idxs: out[i] = (out[i][0] + tx, out[i][1] + ty, out[i][2], out[i][3], out[i][4])
    return out


def pack_regions(polys, canvas, allow_rotate=True, grid_step=5.0, fixed_centers=None):
    return pack_regions_raster_fast(polys, canvas, fixed_centers=fixed_centers, grid_step=max(2.0, float(grid_step)), rotations=[0.0,90.0,180.0,270.0] if allow_rotate else [0.0], search_stride=1, safety_padding=max(1.0, float(getattr(config, "PADDING", 0.0)), float(getattr(config, "PACK_BLEED", 0.0)) * 0.5))


def pack_regions_raster_fast(polys, canvas, fixed_centers=None, grid_step=4.0, rotations=None, search_stride=1, safety_padding=3.0, place_ids=None):
    w, h, safety = canvas[0], canvas[1], max(0.0, float(safety_padding))
    cell = max(1, int(round(float(grid_step))))
    x_min, y_min = int(config.PACK_MARGIN_X)+int(ceil(safety)), int(config.PACK_MARGIN_Y)+int(ceil(safety))
    bw, bh = max(1, int((w-2*config.PACK_MARGIN_X-2*safety)/cell)), max(1, int((h-2*config.PACK_MARGIN_Y-2*safety)/cell))
    grid = np.zeros((bh, bw), dtype=np.uint8)
    rotations = rotations or [0.0, 90.0, 180.0, 270.0]
    placements, rot_info, order, masks, centers, areas = (
        [(-1, -1, 0, 0, False)] * len(polys),
        [{"bin": -1} for _ in range(len(polys))],
        [],
        {},
        {},
        [],
    )
    for rid in (place_ids or range(len(polys))):
        if rid>=len(polys) or len(polys[rid])<3: continue
        pg = Polygon(polys[rid])
        areas.append((rid, pg.area))
        cx, cy = (fixed_centers[rid] if fixed_centers and rid<len(fixed_centers) else pg.centroid.coords[0])
        centers[rid] = (cx, cy)
        for ang in rotations:
            rpts = _rotate_pts(polys[rid], ang, cx, cy)
            xs, ys = [p[0] for p in rpts], [p[1] for p in rpts]
            minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
            mw, mh = int(ceil((maxx-minx)/cell)), int(ceil((maxy-miny)/cell))
            if mw>bw or mh>bh: continue
            img = Image.new("1", (mw, mh), 0); ImageDraw.Draw(img).polygon([((p[0]-minx)/cell, (p[1]-miny)/cell) for p in rpts], fill=1, outline=1)
            m = np.array(img, dtype=np.uint8)
            sc = int(ceil(safety/cell))
            if sc > 0:
                for _ in range(sc):
                    p = np.pad(m, ((1,1),(1,1)), mode="constant")
                    m = (p[1:-1,1:-1]|p[:-2,1:-1]|p[2:,1:-1]|p[1:-1,:-2]|p[1:-1,2:]).astype(np.uint8)
            masks[(rid, ang)] = (m, minx, miny, mw, mh, maxx-minx, maxy-miny)
    for rid in [r for r, _ in sorted(areas, key=lambda t: t[1], reverse=True)]:
        best = None
        for ang in sorted([a for a in rotations if (rid, a) in masks], key=lambda a: (masks[(rid,a)][4], -masks[(rid,a)][3])):
            m, minx, miny, mw, mh, ww, hh = masks[(rid, ang)]
            for y in range(0, bh-mh+1, search_stride):
                for x in range(0, bw-mw+1, search_stride):
                    if not np.any(grid[y:y+mh, x:x+mw] & m):
                        if best is None or (y,x) < (best[2], best[1]): best = (ang, x, y, minx, miny, mw, mh, ww, hh, m)
                        break
                if best and best[0] == ang: break
        if best:
            ang, x, y, minx, miny, mw, mh, ww, hh, m = best
            grid[y:y+mh, x:x+mw] |= m
            placements[rid] = (int(round(x_min+x*cell-minx)), int(round(y_min+y*cell-miny)), int(ceil(ww)), int(ceil(hh)), False)
            rot_info[rid] = {"angle": ang, "cx": centers[rid][0], "cy": centers[rid][1], "minx": minx, "miny": miny, "bin": 0}
            order.append(rid)
    return placements, order, rot_info


def compact_nesting_polygons(polys, placements, rot_info, canvas, step=1.0, passes=2):
    w, h, out = canvas[0], canvas[1], list(placements)
    def _get_poly(rid):
        pts, p, info = polys[rid], out[rid], rot_info[rid]
        tpts = [(pt[0]+p[0], pt[1]+p[1]) for pt in _rotate_pts(pts, info["angle"], info["cx"], info["cy"])]
        return make_valid(Polygon(tpts))
    transformed = [(_get_poly(i) if out[i][2]>0 else None) for i in range(len(out))]
    for _ in range(passes):
        moved = False
        for rid in sorted([i for i, p in enumerate(transformed) if p], key=lambda i: (transformed[i].bounds[1], transformed[i].bounds[0])):
            poly = transformed[rid]
            for dx, dy in [(-step, 0), (0, -step)]:
                while True:
                    cand = _stranslate(poly, xoff=dx, yoff=dy)
                    b = cand.bounds
                    if b[0]<0 or b[1]<0 or b[2]>w or b[3]>h or any(cand.intersects(transformed[j]) and not cand.touches(transformed[j]) for j in range(len(out)) if j!=rid and transformed[j]): break
                    poly, moved = cand, True
                    out[rid] = (int(round(out[rid][0]+dx)), int(round(out[rid][1]+dy)), out[rid][2], out[rid][3], out[rid][4])
            transformed[rid] = poly
        if not moved: break
    return out


def raster_overlap_report(zone_polys, placements, rot_info, canvas, cell=1):
    if not zone_polys or not placements: return {"count": 0, "pairs": []}
    gw, gh, owners, pairs = int(ceil(canvas[0]/cell)), int(ceil(canvas[1]/cell)), {}, set()
    for rid in range(min(len(zone_polys), len(placements))):
        pts, pl, info = zone_polys[rid], placements[rid], rot_info[rid]
        if pl[2]<=0 or not pts: continue
        tpts = [(p[0]+pl[0], p[1]+pl[1]) for p in _rotate_pts(pts, info["angle"], info["cx"], info["cy"])]
        xs, ys = [p[0] for p in tpts], [p[1] for p in tpts]
        x0, y0, x1, y1 = max(0, int(min(xs)/cell)), max(0, int(min(ys)/cell)), min(gw, int(ceil(max(xs)/cell))), min(gh, int(ceil(max(ys)/cell)))
        if x1<=x0 or y1<=y0: continue
        img = Image.new("1", (x1-x0, y1-y0), 0); ImageDraw.Draw(img).polygon([((p[0]/cell)-x0, (p[1]/cell)-y0) for p in tpts], fill=1)
        mask, b = np.array(img, dtype=np.uint8)>0, int(info.get("bin", 0))
        if b not in owners: owners[b] = np.full((gh, gw), -1, dtype=np.int32)
        sub = owners[b][y0:y1, x0:x1]
        hit = sub[mask]
        if hit.size:
            for oid in np.unique(hit[hit>=0]): pairs.add((int(min(oid, rid)), int(max(oid, rid))))
        sub[mask] = rid
    return {"count": len(pairs), "pairs": [list(p) for p in sorted(pairs)]}


def write_pack_svg(
    polys,
    zone_id,
    zone_order,
    zone_polys,
    placements,
    canvas,
    colors,
    rot_info,
    *,
    placement_bin=None,
    placement_bin_by_zid=None,
    page_idx=0,
    out_path=None,
    include_bleed=True,
    write_file=True,
):
    w, h, zone_shift, zone_rot, zone_center, zone_poly_by_zid = canvas[0], canvas[1], {}, {}, {}, {}
    bleed_canvas = float(config.PACK_BLEED)
    for idx, zid in enumerate(zone_order):
        if idx>=len(placements): continue
        p_bin = placement_bin[idx] if placement_bin and idx<len(placement_bin) else (placement_bin_by_zid.get(zid, 0) if placement_bin_by_zid else 0)
        if p_bin != page_idx: continue
        zone_shift[zid], zone_rot[zid], info = (placements[idx][0], placements[idx][1]), rot_info[idx]["angle"], rot_info[idx]
        zone_center[zid] = (info["cx"], info["cy"])
        if idx < len(zone_polys): zone_poly_by_zid[int(zid)] = list(zone_polys[idx])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', f'<rect width="{w}" height="{h}" fill="#1a1a1a" stroke="none"/>', '<g id="fill">']
    for rid, pts in enumerate(polys):
        zid = zone_id[rid] if rid<len(zone_id) else -1
        if zid in zone_shift:
            moved = [(p[0]+zone_shift[zid][0], p[1]+zone_shift[zid][1]) for p in _rotate_pts(pts, zone_rot[zid], zone_center[zid][0], zone_center[zid][1])]
            b, g, r = colors[rid]
            parts.append(f'<path d="M {" L ".join(f"{p[0]:.3f} {p[1]:.3f}" for p in moved)} Z" fill="rgb({r},{g},{b})" stroke="none"/>')
    parts.append('</g><g id="bleed">')
    if include_bleed and bleed_canvas > 0:
        zone_regions = {}
        for rid, zid in enumerate(zone_id): zone_regions.setdefault(int(zid), []).append(rid)
        for zid in zone_order:
            if zid not in zone_shift or zid not in zone_regions: continue
            src_zone = zone_poly_by_zid.get(zid) or []
            if len(src_zone)<3: continue
            zone_base = src_zone[:-1] if len(src_zone)>1 and abs(src_zone[0][0]-src_zone[-1][0])<1e-6 else src_zone
            dx, dy, ang, (cx, cy) = zone_shift[zid][0], zone_shift[zid][1], zone_rot[zid], zone_center[zid]
            boundary = [(p[0]+dx, p[1]+dy) for p in _rotate_pts(zone_base, ang, cx, cy)]
            if len(boundary)<3: continue
            boundary_off = _offset_outline_same_vertices(boundary, bleed_canvas)
            if len(boundary_off)!=len(boundary): continue
            edge_len, cum = [], [0.0]
            for i in range(len(boundary)):
                ll = float(np.hypot(boundary[(i+1)%len(boundary)][0]-boundary[i][0], boundary[(i+1)%len(boundary)][1]-boundary[i][1]))
                edge_len.append(ll); cum.append(cum[-1]+ll)
            def _pt_at(edge_idx, t, seq):
                i0, tt = max(0, min(len(boundary)-1, int(edge_idx))), max(0.0, min(1.0, float(t)))
                return (seq[i0][0]+(seq[(i0+1)%len(boundary)][0]-seq[i0][0])*tt, seq[i0][1]+(seq[(i0+1)%len(boundary)][1]-seq[i0][1])*tt)
            def _project_param(pt):
                best_d2, res = 1e18, None
                for i in range(len(boundary)):
                    a, b = boundary[i], boundary[(i+1)%len(boundary)]
                    vx, vy = b[0]-a[0], b[1]-a[1]
                    l2 = vx*vx+vy*vy
                    if l2<1e-12: continue
                    t = max(0.0, min(1.0, ((pt[0]-a[0])*vx + (pt[1]-a[1])*vy)/l2))
                    px, py = a[0]+vx*t, a[1]+vy*t
                    d2 = (px-pt[0])**2 + (py-pt[1])**2
                    if d2<best_d2: best_d2, res = d2, {"x":px, "y":py, "edge":i, "t":t, "s":cum[i]+edge_len[i]*t, "d2":d2}
                return res
            region_items = []
            for rid in zone_regions[zid]:
                moved_r = [(p[0]+dx, p[1]+dy) for p in _rotate_pts(polys[rid], ang, cx, cy)]
                try:
                    rp = make_valid(Polygon(moved_r))
                    if not rp.is_empty: region_items.append((rid, rp, moved_r))
                except Exception: pass
            if not region_items: continue
            nodes = [{"kind":"v", "edge":i, "t":0.0, "s":cum[i], "x":boundary[i][0], "y":boundary[i][1]} for i in range(len(boundary))]
            for rid, _rp, pts_r in region_items:
                for p in pts_r:
                    pr = _project_param(p)
                    if pr and pr["d2"]<=bleed_canvas**2+1e-9: pr["kind"]="p"; pr["rid"]=rid; nodes.append(pr)
            merged, ordered = [], sorted(nodes, key=lambda e: e["s"])
            for item in ordered:
                if not merged: merged.append(item); continue
                if abs(item["s"]-merged[-1]["s"])<=0.5:
                    if merged[-1]["kind"]!="v" and item["kind"]=="v": merged[-1]=item
                else: merged.append(item)
            for i in range(len(merged)):
                a, b = merged[i], merged[(i+1)%len(merged)]
                aox, aoy = _pt_at(a["edge"], a["t"], boundary_off); box, boy = _pt_at(b["edge"], b["t"], boundary_off)
                bleed_poly = [(a["x"], a["y"]), (aox, aoy), (box, boy), (b["x"], b["y"])]
                mid = Point(0.5*(a["x"]+b["x"]), 0.5*(a["y"]+b["y"]))
                best_rid, best_d = None, 1e18
                for rid, rp, _ in region_items:
                    try:
                        d = rp.boundary.distance(mid)
                        if d<best_d: best_d, best_rid = d, rid
                    except Exception: pass
                bb, bg, br = colors[best_rid] if best_rid is not None else (200,200,200)
                parts.append(f'<path d="M {" L ".join(f"{p[0]:.3f} {p[1]:.3f}" for p in bleed_poly)} Z" fill="rgb({br},{bg},{bb})" stroke="none"/>')
    parts.append("</g></svg>")
    svg_text = "".join(parts)
    if write_file:
        (out_path or config.OUT_PACK_SVG).write_text(svg_text, encoding="utf-8")
    return svg_text


def compute_scene(svg_path, snap: float, include_packed: bool = False) -> Dict:
    config._apply_pack_env()
    regions, polys, canvas, debug = geometry.build_regions_from_svg(svg_path, snap_override=snap)
    zone_id = zones.build_zones(polys, config.TARGET_ZONES)
    zone_id, zone_members = zones._remap_zones_by_area(polys, zone_id)
    zone_boundaries = zones.build_zone_boundaries(polys, zone_id)
    zone_geoms = zones.build_zone_geoms(polys, zone_id)
    zone_polys, zone_order, zone_poly_debug = zones.build_zone_polys(polys, zone_id)
    zone_pack_polys = _build_zone_pack_polys(zone_polys, float(config.PACK_BLEED))
    placements = [(-1, -1, 0, 0, False) for _ in zone_order]
    rot_info = [{"angle": 0.0, "cx": 0.0, "cy": 0.0, "minx": 0.0, "miny": 0.0, "bin": -1} for _ in zone_order]
    if include_packed and zone_pack_polys:
        zone_pack_centers = [Polygon(p).centroid.coords[0] if p else (0, 0) for p in zone_pack_polys]
        placements, order, rot_info = pack_regions_raster_fast(
            zone_pack_polys, canvas, fixed_centers=zone_pack_centers
        )
        placements = _resolve_pack_overlaps(
            zone_pack_polys, placements, rot_info, step=1.0, padding=float(config.PADDING)
        )
        placements = compact_nesting_polygons(zone_pack_polys, placements, rot_info, canvas)
        placements = _fit_placements_into_canvas(placements, rot_info, canvas)
    zone_labels, zone_label_map = {}, {z: idx+1 for idx, z in enumerate(zone_order)}
    for zid, geom in zone_geoms.items():
        lx, ly, idx = None, None, {z: i for i, z in enumerate(zone_order)}.get(zid)
        if idx is not None and idx<len(zone_polys) and zone_polys[idx]:
            try:
                c = Polygon(zone_polys[idx]).centroid
                if geom.covers(Point(c.x, c.y)): lx, ly = float(c.x), float(c.y)
            except Exception: pass
        if lx is None:
            m = zone_members.get(zid, [])
            lx, ly = (Polygon(polys[m[0]]).centroid.coords[0] if m else zones._label_pos_for_zone(geom))
        zone_labels[str(zid)] = {"x": lx, "y": ly, "label": zone_label_map.get(zid, zid)}
    region_labels = {str(rid): {"x": float(Polygon(p).centroid.x), "y": float(Polygon(p).centroid.y), "label": rid, "zone": zone_id[rid] if rid<len(zone_id) else -1} for rid, p in enumerate(polys) if len(p)>=3}
    colors, _ = geometry.compute_region_colors(polys, canvas)
    if include_packed:
        write_pack_svg(
            polys,
            zone_id,
            zone_order,
            zone_polys,
            placements,
            canvas,
            colors,
            rot_info,
            page_idx=0,
            out_path=config.OUT_PACK_SVG,
        )
    return {
        "canvas": {"w": canvas[0], "h": canvas[1]}, "draw_scale": config.DRAW_SCALE, "regions": polys, "zone_boundaries": zone_boundaries, "zone_id": zone_id, "zone_labels": zone_labels, "region_labels": region_labels, "zone_order": zone_order, "zone_pack_polys": zone_pack_polys,
        "zone_rot": {zid: rot_info[idx]["angle"] for idx, zid in enumerate(zone_order) if idx < len(rot_info)},
        "zone_center": {zid: (rot_info[idx]["cx"], rot_info[idx]["cy"]) for idx, zid in enumerate(zone_order) if idx < len(rot_info)},
        "zone_shift": {zid: (placements[idx][0], placements[idx][1]) for idx, zid in enumerate(zone_order) if idx < len(placements)},
        "placement_bin": {zid: int(rot_info[idx].get("bin", 0)) for idx, zid in enumerate(zone_order) if idx < len(rot_info)},
        "region_colors": [f"#{r:02x}{g:02x}{b:02x}" for (b, g, r) in colors],
        "colors_bgr": [[int(b), int(g), int(r)] for (b, g, r) in colors],
        "placements": [[int(a), int(b), int(c), int(d), bool(e)] for (a, b, c, d, e) in placements],
        "rot_info": [{"angle": float(i["angle"]), "cx": float(i["cx"]), "cy": float(i["cy"])} for i in rot_info],
        "debug": debug, "snap": snap,
    }

if __name__ == "__main__":
    if config.SVG_PATH.exists(): compute_scene(config.SVG_PATH, 0.5)
