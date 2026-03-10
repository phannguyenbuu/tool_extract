from __future__ import annotations

import json
import math
import sys
from html import escape
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import config, packing, source_voronoi, zones  # noqa: E402


OUT_DIR = ROOT / "tmp_svgs"
MAX_PROJ_DIST = 4.0
PANEL_PAD = 18.0
CELL_W = 280.0
CELL_H = 220.0
COLS = 3
DEBUG_BLEED = 3.0


def active_source_path() -> Path:
    active_json = ROOT / "active_source.json"
    if active_json.exists():
        try:
            name = json.loads(active_json.read_text(encoding="utf-8")).get("name")
            if name:
                for candidate in (ROOT / "sources" / name, ROOT / name):
                    if candidate.exists():
                        return candidate
        except Exception:
            pass
    for candidate in (ROOT / "sources" / "chobenthanh.svg", ROOT / "chobenthanh.svg"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No active source SVG found")


def poly_open(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(poly) > 1 and poly[0] == poly[-1]:
        return poly[:-1]
    return poly[:]


def bbox_of(items: list[list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    pts = [pt for poly in items for pt in poly]
    if not pts:
        return (0.0, 0.0, 100.0, 100.0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def fit_transform(
    bbox: tuple[float, float, float, float],
    cell_x: float,
    cell_y: float,
    cell_w: float,
    cell_h: float,
) -> tuple[float, float, float]:
    minx, miny, maxx, maxy = bbox
    w = max(maxx - minx, 1.0)
    h = max(maxy - miny, 1.0)
    scale = min((cell_w - 2 * PANEL_PAD) / w, (cell_h - 2 * PANEL_PAD) / h)
    dx = cell_x + (cell_w - w * scale) * 0.5 - minx * scale
    dy = cell_y + (cell_h - h * scale) * 0.5 - miny * scale
    return scale, dx, dy


def tr(pt: tuple[float, float], scale: float, dx: float, dy: float) -> tuple[float, float]:
    return (pt[0] * scale + dx, pt[1] * scale + dy)


def draw_poly(poly: list[tuple[float, float]], scale: float, dx: float, dy: float, stroke: str, width: float, fill: str = "none") -> str:
    pts = poly_open(poly)
    if len(pts) < 2:
        return ""
    d = "M " + " L ".join(f"{tr(p, scale, dx, dy)[0]:.3f} {tr(p, scale, dx, dy)[1]:.3f}" for p in pts) + " Z"
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" />'


def draw_polys(
    polys: list[list[tuple[float, float]]],
    scale: float,
    dx: float,
    dy: float,
    stroke: str,
    width: float,
    fill: str,
    fill_opacity: float = 0.2,
) -> str:
    out: list[str] = []
    for poly in polys:
        pts = poly_open(poly)
        if len(pts) < 3:
            continue
        d = "M " + " L ".join(f"{tr(p, scale, dx, dy)[0]:.3f} {tr(p, scale, dx, dy)[1]:.3f}" for p in pts) + " Z"
        out.append(
            f'<path d="{d}" fill="{fill}" fill-opacity="{fill_opacity}" stroke="{stroke}" stroke-width="{width}" />'
        )
    return "".join(out)


def draw_polys_multicolor(
    polys: list[list[tuple[float, float]]],
    scale: float,
    dx: float,
    dy: float,
    width: float = 0.9,
    fill_opacity: float = 0.44,
) -> str:
    palette = [
        "#ff6b6b",
        "#ffd166",
        "#06d6a0",
        "#4cc9f0",
        "#a78bfa",
        "#f72585",
        "#f9844a",
        "#43aa8b",
        "#90be6d",
        "#577590",
    ]
    out: list[str] = []
    for idx, poly in enumerate(polys):
        pts = poly_open(poly)
        if len(pts) < 3:
            continue
        color = palette[idx % len(palette)]
        d = "M " + " L ".join(f"{tr(p, scale, dx, dy)[0]:.3f} {tr(p, scale, dx, dy)[1]:.3f}" for p in pts) + " Z"
        out.append(
            f'<path d="{d}" fill="{color}" fill-opacity="{fill_opacity}" stroke="{color}" stroke-width="{width}" />'
        )
    return "".join(out)


def draw_polys_with_colors(
    polys: list[list[tuple[float, float]]],
    colors: list[str],
    scale: float,
    dx: float,
    dy: float,
    width: float = 0.9,
    fill_opacity: float = 0.44,
) -> str:
    out: list[str] = []
    for idx, poly in enumerate(polys):
        pts = poly_open(poly)
        if len(pts) < 3:
            continue
        color = colors[idx] if idx < len(colors) else "#9ca3af"
        d = "M " + " L ".join(f"{tr(p, scale, dx, dy)[0]:.3f} {tr(p, scale, dx, dy)[1]:.3f}" for p in pts) + " Z"
        out.append(
            f'<path d="{d}" fill="{color}" fill-opacity="{fill_opacity}" stroke="{color}" stroke-width="{width}" />'
        )
    return "".join(out)


def draw_point_labels(
    pts: list[tuple[float, float]],
    scale: float,
    dx: float,
    dy: float,
    prefix: str,
    color: str,
    radius: float = 2.6,
) -> str:
    out: list[str] = []
    for idx, pt in enumerate(poly_open(pts)):
        x, y = tr(pt, scale, dx, dy)
        out.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius}" fill="{color}" />')
        out.append(
            f'<text x="{x + 4:.3f}" y="{y - 4:.3f}" fill="{color}" font-size="10">{escape(f"{prefix}{idx}")}</text>'
        )
    return "".join(out)


def draw_projection_panel(
    outline: list[tuple[float, float]],
    projections: list[tuple[tuple[float, float], tuple[float, float], int]],
    scale: float,
    dx: float,
    dy: float,
) -> str:
    out = [draw_poly(outline, scale, dx, dy, "#8fa3bf", 1.2)]
    for order, (src_pt, proj_pt, _seg_idx) in enumerate(projections):
        sx, sy = tr(src_pt, scale, dx, dy)
        px, py = tr(proj_pt, scale, dx, dy)
        out.append(f'<line x1="{sx:.3f}" y1="{sy:.3f}" x2="{px:.3f}" y2="{py:.3f}" stroke="#4a5568" stroke-width="0.8" />')
        out.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="2.8" fill="#ffd400" />')
        out.append(
            f'<text x="{px + 4:.3f}" y="{py - 4:.3f}" fill="#ffd400" font-size="10">{escape(f"p{order}")}</text>'
        )
    return "".join(out)


def draw_ordered_points_panel(
    outline: list[tuple[float, float]],
    ordered_points: list[dict[str, object]],
    scale: float,
    dx: float,
    dy: float,
) -> str:
    out = [draw_poly(outline, scale, dx, dy, "#8fa3bf", 1.2)]
    for order, item in enumerate(ordered_points):
        pt = item["pt"]
        tag = str(item["tag"])
        px, py = tr(pt, scale, dx, dy)
        color = "#ffffff" if tag.startswith("v") else "#ffd400"
        out.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="2.8" fill="{color}" />')
        out.append(
            f'<text x="{px + 4:.3f}" y="{py - 4:.3f}" fill="{color}" font-size="10">{escape(f"{order}:{tag}")}</text>'
        )
    return "".join(out)


def draw_marked_points_panel(
    outline: list[tuple[float, float]],
    points: list[dict[str, object]],
    scale: float,
    dx: float,
    dy: float,
) -> str:
    out = [draw_poly(outline, scale, dx, dy, "#8fa3bf", 1.2)]
    for idx, item in enumerate(points):
        pt = item["pt"]
        rid = int(item["region_id"])
        vid = int(item["vertex_index"])
        dist = float(item["dist"])
        px, py = tr(pt, scale, dx, dy)
        out.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="2.8" fill="#22d3ee" />')
        out.append(
            f'<text x="{px + 4:.3f}" y="{py - 4:.3f}" fill="#22d3ee" font-size="10">{escape(f"{idx}:r{rid}v{vid} d={dist:.2f}")}</text>'
        )
    return "".join(out)


def project_vertices_to_zone(
    source_lines: list[LineString],
    zone_poly: list[tuple[float, float]],
    max_dist: float,
) -> list[tuple[tuple[float, float], tuple[float, float], int]]:
    zone_pts = poly_open(zone_poly)
    segs: list[tuple[int, tuple[float, float], tuple[float, float]]] = []
    for idx, (a, b) in enumerate(zip(zone_pts, zone_pts[1:] + zone_pts[:1])):
        if a != b:
            segs.append((idx, a, b))

    seen: set[tuple[int, int, int, int]] = set()
    projected: list[tuple[tuple[float, float], tuple[float, float], int, float]] = []
    boundary = LineString(zone_pts + [zone_pts[0]])
    for line in source_lines:
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        for raw in (coords[0], coords[-1]):
            src_pt = (float(raw[0]), float(raw[1]))
            best = None
            best_d2 = max_dist * max_dist
            best_seg = -1
            for seg_idx, a, b in segs:
                proj = source_voronoi._project_point_to_segment(src_pt, a, b)
                if proj is None:
                    continue
                dx = proj[0] - src_pt[0]
                dy = proj[1] - src_pt[1]
                d2 = dx * dx + dy * dy
                if d2 <= best_d2:
                    best_d2 = d2
                    best = proj
                    best_seg = seg_idx
            if best is None:
                continue
            key = (
                round(src_pt[0] * 1000),
                round(src_pt[1] * 1000),
                round(best[0] * 1000),
                round(best[1] * 1000),
            )
            if key in seen:
                continue
            seen.add(key)
            proj_pos = float(boundary.project(Point(best)))
            projected.append((src_pt, best, best_seg, proj_pos))

    projected.sort(key=lambda item: item[3])
    return [(src, proj, seg_idx) for src, proj, seg_idx, _ in projected]


def panel_title(title: str, x: float, y: float) -> str:
    return f'<text x="{x + 8:.3f}" y="{y + 16:.3f}" fill="#ffffff" font-size="13">{escape(title)}</text>'


def build_bleed_polygons(
    outline: list[tuple[float, float]], bleed: float
) -> list[dict[str, object]]:
    pts = poly_open(outline)
    if len(pts) < 3 or bleed <= 0:
        return []
    ocoords = packing._offset_outline_same_vertices(pts, bleed)
    npts = len(pts)
    out: list[dict[str, object]] = []
    for i in range(npts):
        p0 = pts[i]
        p1 = pts[(i + 1) % npts]
        p0o = ocoords[i]
        p1o = ocoords[(i + 1) % npts]
        ax, ay = p0
        bx, by = p1
        aox, aoy = p0o
        box, boy = p1o
        abx = box - aox
        aby = boy - aoy
        ab_len2 = abx * abx + aby * aby
        if ab_len2 <= 1e-8:
            continue
        t_a = ((ax - aox) * abx + (ay - aoy) * aby) / ab_len2
        t_b = ((bx - aox) * abx + (by - aoy) * aby) / ab_len2
        t_a = max(0.0, min(1.0, t_a))
        t_b = max(0.0, min(1.0, t_b))
        a1x = aox + t_a * abx
        a1y = aoy + t_a * aby
        b1x = aox + t_b * abx
        b1y = aoy + t_b * aby
        dax = aox - ax
        day = aoy - ay
        dbx = box - bx
        dby = boy - by
        da_len = math.hypot(dax, day)
        db_len = math.hypot(dbx, dby)
        if da_len > 1e-6:
            s = min(1.0, bleed / da_len)
            a2x = ax + dax * s
            a2y = ay + day * s
        else:
            a2x, a2y = ax, ay
        if db_len > 1e-6:
            s = min(1.0, bleed / db_len)
            b2x = bx + dbx * s
            b2y = by + dby * s
        else:
            b2x, b2y = bx, by
        out.append(
            {
                "poly": [
                    (ax, ay),
                    (a2x, a2y),
                    (a1x, a1y),
                    (b1x, b1y),
                    (b2x, b2y),
                    (bx, by),
                ],
                "edge": ((ax, ay), (bx, by)),
            }
        )
    return out


def build_vertex_order(
    zone_poly: list[tuple[float, float]],
    projections: list[tuple[tuple[float, float], tuple[float, float], int]],
) -> list[dict[str, object]]:
    pts = poly_open(zone_poly)
    boundary = LineString(pts + [pts[0]])
    out: list[dict[str, object]] = []
    for vidx, pt in enumerate(pts):
        s = float(boundary.project(Point(pt)))
        out.append({"tag": f"v{vidx}", "pt": pt, "s": s})
    for pidx, (_src, proj_pt, _seg_idx) in enumerate(projections):
        s = float(boundary.project(Point(proj_pt)))
        out.append({"tag": f"p{pidx}", "pt": proj_pt, "s": s})
    out.sort(key=lambda item: float(item["s"]))
    return out


def merge_vertex_order(
    ordered_points: list[dict[str, object]],
    threshold: float,
) -> list[dict[str, object]]:
    if not ordered_points:
        return []
    merged: list[dict[str, object]] = []
    i = 0
    while i < len(ordered_points):
        cluster = [ordered_points[i]]
        j = i + 1
        prev_pt = ordered_points[i]["pt"]
        while j < len(ordered_points):
            cur_pt = ordered_points[j]["pt"]
            dist = math.hypot(cur_pt[0] - prev_pt[0], cur_pt[1] - prev_pt[1])
            if dist > threshold:
                break
            cluster.append(ordered_points[j])
            prev_pt = cur_pt
            j += 1
        cx = sum(item["pt"][0] for item in cluster) / len(cluster)
        cy = sum(item["pt"][1] for item in cluster) / len(cluster)
        tags = "+".join(str(item["tag"]) for item in cluster)
        merged.append({"tag": tags, "pt": (cx, cy), "s": float(cluster[0]["s"])})
        i = j
    return merged


def ordered_points_to_poly(ordered_points: list[dict[str, object]]) -> list[tuple[float, float]]:
    return [item["pt"] for item in ordered_points if "pt" in item]


def _merge_ordered_boundary_points(
    ordered: list[tuple[float, tuple[float, float]]],
    threshold: float,
) -> list[tuple[float, float]]:
    if not ordered:
        return []
    if threshold <= 0:
        return [pt for _s, pt in ordered]
    out: list[tuple[float, float]] = []
    i = 0
    while i < len(ordered):
        cluster = [ordered[i][1]]
        j = i + 1
        prev = ordered[i][1]
        while j < len(ordered):
            cur = ordered[j][1]
            if math.hypot(cur[0] - prev[0], cur[1] - prev[1]) > threshold:
                break
            cluster.append(cur)
            prev = cur
            j += 1
        cx = sum(p[0] for p in cluster) / len(cluster)
        cy = sum(p[1] for p in cluster) / len(cluster)
        out.append((cx, cy))
        i = j
    return out


def _weld_projected_points(
    ordered: list[tuple[float, tuple[float, float]]],
    zone_outline: list[tuple[float, float]],
    weld_threshold: float,
    vertex_snap_threshold: float,
) -> list[tuple[float, float]]:
    if not ordered:
        return []
    welded = _merge_ordered_boundary_points(ordered, weld_threshold)
    if not welded:
        return []

    snapped: list[tuple[float, float]] = []
    v2 = vertex_snap_threshold * vertex_snap_threshold
    for p in welded:
        best = p
        best_d2 = v2
        for v in zone_outline:
            dx = p[0] - v[0]
            dy = p[1] - v[1]
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best_d2 = d2
                best = (float(v[0]), float(v[1]))
        snapped.append(best)

    deduped: list[tuple[float, float]] = []
    for p in snapped:
        if not deduped or math.hypot(p[0] - deduped[-1][0], p[1] - deduped[-1][1]) > 1e-9:
            deduped.append(p)
    if len(deduped) > 1 and math.hypot(deduped[0][0] - deduped[-1][0], deduped[0][1] - deduped[-1][1]) <= 1e-9:
        deduped = deduped[:-1]
    return deduped


def build_outline_from_near_vertices(
    near_vertices: list[dict[str, object]],
    zone_outline: list[tuple[float, float]],
    merge_threshold: float = 0.5,
) -> list[tuple[float, float]]:
    if len(zone_outline) < 3 or not near_vertices:
        return []
    boundary = LineString(zone_outline + [zone_outline[0]])
    ordered: list[tuple[float, tuple[float, float]]] = []
    for item in near_vertices:
        p = item["pt"]
        s = float(boundary.project(Point(p)))
        pi = boundary.interpolate(s)
        ordered.append((s, (float(pi.x), float(pi.y))))
    ordered.sort(key=lambda x: x[0])
    welded = _weld_projected_points(
        ordered,
        zone_outline,
        weld_threshold=1.0,
        vertex_snap_threshold=1.0,
    )
    if merge_threshold > 0:
        ordered2 = [(float(i), p) for i, p in enumerate(welded)]
        return _merge_ordered_boundary_points(ordered2, merge_threshold)
    return welded


def _point_segment_dist2(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    vx = b[0] - a[0]
    vy = b[1] - a[1]
    ll = vx * vx + vy * vy
    if ll <= 1e-12:
        dx = p[0] - a[0]
        dy = p[1] - a[1]
        return dx * dx + dy * dy
    t = ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / ll
    t = max(0.0, min(1.0, t))
    qx = a[0] + t * vx
    qy = a[1] + t * vy
    dx = p[0] - qx
    dy = p[1] - qy
    return dx * dx + dy * dy


def assign_bleed_colors_from_regions(
    bleed_items: list[dict[str, object]],
    region_pts: list[list[tuple[float, float]]],
    region_colors: list[str],
) -> list[str]:
    if not bleed_items:
        return []
    if not region_pts:
        return ["#9ca3af"] * len(bleed_items)
    out: list[str] = []
    for item in bleed_items:
        a, b = item["edge"]
        mid = ((float(a[0]) + float(b[0])) * 0.5, (float(a[1]) + float(b[1])) * 0.5)
        best_idx = -1
        best_d2 = None
        for ridx, poly in enumerate(region_pts):
            if len(poly) < 2:
                continue
            for p0, p1 in zip(poly, poly[1:] + poly[:1]):
                d2 = _point_segment_dist2(mid, p0, p1)
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_idx = ridx
        color = "#9ca3af"
        if 0 <= best_idx < len(region_colors):
            color = str(region_colors[best_idx])
        out.append(color)
    return out


def clip_regions_to_zone(
    region_ids: list[int],
    region_pts: list[list[tuple[float, float]]],
    zone_outline: list[tuple[float, float]],
) -> tuple[list[int], list[list[tuple[float, float]]]]:
    if len(zone_outline) < 3:
        return region_ids, region_pts
    try:
        zone_poly = Polygon(zone_outline)
    except Exception:
        return region_ids, region_pts
    if zone_poly.is_empty:
        return region_ids, region_pts
    out_ids: list[int] = []
    out_pts: list[list[tuple[float, float]]] = []
    for rid, pts in zip(region_ids, region_pts):
        if len(pts) < 3:
            continue
        try:
            rp = Polygon(pts)
        except Exception:
            continue
        if rp.is_empty:
            continue
        inter = rp.intersection(zone_poly)
        if inter.is_empty:
            continue
        if inter.geom_type == "Polygon":
            coords = list(inter.exterior.coords)
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) >= 3:
                out_ids.append(rid)
                out_pts.append([(float(x), float(y)) for x, y in coords])
        else:
            geoms = [g for g in getattr(inter, "geoms", []) if g.geom_type == "Polygon"]
            if not geoms:
                continue
            best = max(geoms, key=lambda g: float(g.area))
            coords = list(best.exterior.coords)
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) >= 3:
                out_ids.append(rid)
                out_pts.append([(float(x), float(y)) for x, y in coords])
    return out_ids, out_pts


def find_near_boundary_vertices(
    region_ids: list[int],
    region_pts: list[list[tuple[float, float]]],
    outline: list[tuple[float, float]],
    radius: float,
) -> list[dict[str, object]]:
    if len(outline) < 2 or radius < 0:
        return []
    boundary = LineString(outline + [outline[0]])
    hits: list[dict[str, object]] = []
    for rid, poly in zip(region_ids, region_pts):
        for vid, pt in enumerate(poly):
            d = float(boundary.distance(Point(pt)))
            if d <= radius + 1e-9:
                hits.append(
                    {
                        "region_id": int(rid),
                        "vertex_index": int(vid),
                        "pt": (float(pt[0]), float(pt[1])),
                        "dist": d,
                    }
                )
    hits.sort(key=lambda it: (float(it["dist"]), int(it["region_id"]), int(it["vertex_index"])))
    return hits


def export_zone(
    zone_id: int,
    vor: dict,
    region_scene: dict,
    source_lines: list[LineString],
) -> Path:
    snapped_cells = vor.get("snapped_cells", []) or []
    if zone_id >= len(snapped_cells):
        raise RuntimeError(f"Zone {zone_id} missing in snapped_cells")
    zone_original = [(float(x), float(y)) for x, y in snapped_cells[zone_id]]

    zone_region_ids = [int(rid) for rid in region_scene.get("snap_region_map", {}).get(str(zone_id), [])]
    zone_region_pts = [
        [(float(x), float(y)) for x, y in region_scene["regions"][rid]]
        for rid in zone_region_ids
        if 0 <= rid < len(region_scene.get("regions", []))
    ]
    zone_outline_map = zones.build_zone_boundaries(zone_region_pts, [0] * len(zone_region_pts))
    zone_outline = poly_open((zone_outline_map.get(0) or [[]])[0])
    zone_region_ids, zone_region_pts = clip_regions_to_zone(zone_region_ids, zone_region_pts, zone_outline)
    all_region_colors = region_scene.get("region_colors", []) or []
    zone_region_colors = [
        str(all_region_colors[rid]) if 0 <= rid < len(all_region_colors) else "#9ca3af"
        for rid in zone_region_ids
    ]

    projections = project_vertices_to_zone(source_lines, zone_original, MAX_PROJ_DIST)
    vertex_order = build_vertex_order(zone_original, projections)
    vertex_order_merged = merge_vertex_order(vertex_order, threshold=0.5)
    near_vertices = find_near_boundary_vertices(
        zone_region_ids,
        zone_region_pts,
        zone_outline,
        float(DEBUG_BLEED),
    )
    merged_outline = build_outline_from_near_vertices(
        near_vertices,
        zone_outline,
        merge_threshold=0.5,
    )
    if len(merged_outline) < 3:
        merged_outline = ordered_points_to_poly(vertex_order_merged)

    bleed = DEBUG_BLEED
    bleed_offset = (
        packing._offset_outline_same_vertices(merged_outline, bleed)
        if len(merged_outline) >= 3
        else []
    )
    bleed_items = build_bleed_polygons(merged_outline, bleed)
    bleed_polys = [item["poly"] for item in bleed_items]
    bleed_colors = assign_bleed_colors_from_regions(bleed_items, zone_region_pts, zone_region_colors)

    panels = [
        ("Original Zone", [zone_original]),
        ("Zone Outline", [zone_outline]),
        ("Vertex Order", [zone_original]),
        ("Vertex Order After Merge (th=0.5)", [zone_original]),
        ("Region Vertices <= Bleed", [zone_outline]),
        ("Bleed Polygons", [zone_outline, merged_outline, bleed_offset] + bleed_polys),
        ("Final", [zone_outline, merged_outline, bleed_offset] + zone_region_pts + bleed_polys),
    ]

    width = COLS * CELL_W
    rows = math.ceil(len(panels) / COLS)
    height = rows * CELL_H
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="#0a1020" />',
    ]

    for idx, (title, polys) in enumerate(panels):
        col = idx % COLS
        row = idx // COLS
        x0 = col * CELL_W
        y0 = row * CELL_H
        parts.append(f'<rect x="{x0 + 4:.3f}" y="{y0 + 4:.3f}" width="{CELL_W - 8:.3f}" height="{CELL_H - 8:.3f}" rx="10" fill="#0f172a" stroke="#23314f" stroke-width="1" />')
        parts.append(panel_title(title, x0, y0))
        bbox = bbox_of([poly for poly in polys if poly])
        scale, dx, dy = fit_transform(bbox, x0, y0 + 18.0, CELL_W, CELL_H - 18.0)

        if title == "Original Zone":
            parts.append(draw_poly(zone_original, scale, dx, dy, "#38bdf8", 1.6, "rgba(56,189,248,0.08)"))
        elif title == "Zone Outline":
            parts.append(draw_poly(zone_outline, scale, dx, dy, "#ffffff", 1.3))
        elif title == "Vertex Order":
            parts.append(draw_ordered_points_panel(zone_original, vertex_order, scale, dx, dy))
        elif title == "Vertex Order After Merge (th=0.5)":
            parts.append(draw_ordered_points_panel(zone_original, vertex_order_merged, scale, dx, dy))
        elif title == "Region Vertices <= Bleed":
            parts.append(draw_marked_points_panel(zone_outline, near_vertices, scale, dx, dy))
        elif title == "Bleed Polygons":
            parts.append(draw_poly(zone_outline, scale, dx, dy, "#475569", 0.9))
            parts.append(draw_poly(merged_outline, scale, dx, dy, "#8fa3bf", 1.0))
            parts.append(draw_poly(bleed_offset, scale, dx, dy, "#ffd400", 1.2))
            parts.append(draw_polys_with_colors(bleed_polys, bleed_colors, scale, dx, dy, width=1.0, fill_opacity=0.26))
        elif title == "Final":
            parts.append(draw_polys_with_colors(zone_region_pts, zone_region_colors, scale, dx, dy, width=0.8, fill_opacity=0.36))
            parts.append(draw_poly(zone_outline, scale, dx, dy, "#475569", 0.9))
            parts.append(draw_poly(merged_outline, scale, dx, dy, "#d1d5db", 1.0))
            parts.append(draw_polys_with_colors(bleed_polys, bleed_colors, scale, dx, dy, width=1.0, fill_opacity=0.46))

    parts.append("</svg>")

    out_path = OUT_DIR / f"zone_{zone_id}_projection_grid.svg"
    out_path.write_text("".join(parts), encoding="utf-8")
    near_path = OUT_DIR / f"zone_{zone_id}_near_vertices_bleed.json"
    near_path.write_text(
        json.dumps(
            {
                "zone_id": int(zone_id),
                "bleed_radius": float(DEBUG_BLEED),
                "count": len(near_vertices),
                "vertices": [
                    {
                        "region_id": int(item["region_id"]),
                        "vertex_index": int(item["vertex_index"]),
                        "x": float(item["pt"][0]),
                        "y": float(item["pt"][1]),
                        "dist": float(item["dist"]),
                    }
                    for item in near_vertices
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def parse_zone_ids(argv: list[str]) -> list[int]:
    if len(argv) <= 1:
        return [0]
    out: list[int] = []
    for raw in argv[1:]:
        out.append(int(raw))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = active_source_path()
    vor = source_voronoi.build_source_voronoi(src, count=120, relax=2, seed=7)
    region_scene = source_voronoi.build_source_region_scene(src, count=120, relax=2, seed=7, cached_voronoi=vor)
    source_lines, _canvas = source_voronoi._load_source_segments(src)
    zone_ids = parse_zone_ids(sys.argv)
    for zid in zone_ids:
        export_zone(zid, vor, region_scene, source_lines)


if __name__ == "__main__":
    main()
