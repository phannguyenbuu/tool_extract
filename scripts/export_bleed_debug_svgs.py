from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import config, geometry, packing, zones

OUT_DIR = ROOT / "tmp_svgs"
TARGET_ZONES = [0, 3, 54, 104]


def signed_area(poly: list[tuple[float, float]]) -> float:
    pts = poly[:-1] if len(poly) > 1 and poly[0] == poly[-1] else poly
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for a, b in zip(pts, pts[1:] + pts[:1]):
        area += a[0] * b[1] - b[0] * a[1]
    return area / 2.0


def bbox_of(polys: Iterable[list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    pts = [p for poly in polys for p in poly]
    if not pts:
        return (0.0, 0.0, 100.0, 100.0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def path_d(poly: list[tuple[float, float]]) -> str:
    pts = poly[:-1] if len(poly) > 1 and poly[0] == poly[-1] else poly
    if not pts:
        return ""
    return "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in pts) + " Z"


def draw_poly(poly: list[tuple[float, float]], stroke: str, fill: str = "none", width: float = 1.5) -> str:
    d = path_d(poly)
    if not d:
        return ""
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" />'


def draw_markers(poly: list[tuple[float, float]], prefix: str, color: str) -> str:
    pts = poly[:-1] if len(poly) > 1 and poly[0] == poly[-1] else poly
    if len(pts) < 2:
        return ""
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    parts = [
        f'<circle cx="{x0:.3f}" cy="{y0:.3f}" r="3.5" fill="#00ff7f" stroke="#001b12" stroke-width="1"/>',
        f'<circle cx="{x1:.3f}" cy="{y1:.3f}" r="3.0" fill="#ff9f1c" stroke="#2b1600" stroke-width="1"/>',
        f'<text x="{x0 + 6:.3f}" y="{y0 - 6:.3f}" fill="{color}" font-size="9">{prefix}:start</text>',
        f'<text x="{x1 + 6:.3f}" y="{y1 + 10:.3f}" fill="{color}" font-size="9">{prefix}:next</text>',
    ]
    return "".join(parts)


def source_path() -> Path:
    active_json = ROOT / "active_source.json"
    if active_json.exists():
        try:
            import json

            active = json.loads(active_json.read_text(encoding="utf-8")).get("name")
            if active:
                for candidate in (ROOT / "sources" / active, ROOT / active):
                    if candidate.exists():
                        return candidate
        except Exception:
            pass
    for candidate in (ROOT / "sources" / "chobenthanh.svg", ROOT / "chobenthanh.svg"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No source SVG found")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = source_path()
    regions, polys, _canvas, _debug = geometry.build_regions_from_svg(src)
    zone_id = zones.build_zones(polys, config.TARGET_ZONES)
    zone_id, _members = zones._remap_zones_by_area(polys, zone_id)
    zone_boundaries = zones.build_zone_boundaries(polys, zone_id)
    zone_polys, zone_order, zone_debug = zones.build_zone_polys(polys, zone_id)
    zone_index = {zid: idx for idx, zid in enumerate(zone_order)}
    bleed = float(config.PACK_BLEED)

    summary_lines = [
        f"source={src.name}",
        f"pack_bleed={bleed}",
        f"convex_hull_zones={zone_debug.get('convex_hull', [])}",
    ]

    for zid in TARGET_ZONES:
        idx = zone_index.get(zid)
        if idx is None:
            summary_lines.append(f"zone {zid}: missing from zone_order")
            continue

        zone_poly = zone_polys[idx]
        boundary_paths = zone_boundaries.get(zid, [])
        boundary = boundary_paths[0] if boundary_paths else []
        boundary_open = boundary[:-1] if len(boundary) > 1 and boundary[0] == boundary[-1] else boundary
        offset = packing._offset_outline_same_vertices(boundary_open, bleed) if len(boundary_open) >= 3 else []
        bevel, _dbg = packing._bevel_outline_by_angle(offset, bleed, angle_thresh=60.0) if offset else ([], [])
        pack_poly = packing._build_zone_pack_polys([zone_poly], bleed, bevel_angle=60.0)[0]

        minx, miny, maxx, maxy = bbox_of([p for p in [boundary_open, offset, bevel, zone_poly, pack_poly] if p])
        pad = 20.0
        view_x = minx - pad
        view_y = miny - pad
        view_w = max(120.0, (maxx - minx) + pad * 2)
        view_h = max(120.0, (maxy - miny) + pad * 2)

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_x:.3f} {view_y:.3f} {view_w:.3f} {view_h:.3f}" width="{view_w:.0f}" height="{view_h:.0f}">',
            '<rect width="100%" height="100%" fill="#081226"/>',
            f'<text x="{view_x + 6:.3f}" y="{view_y + 14:.3f}" fill="#ffffff" font-size="11">zone {zid}</text>',
            f'<text x="{view_x + 6:.3f}" y="{view_y + 28:.3f}" fill="#7dd3fc" font-size="10">zone_poly area={signed_area(zone_poly):.3f}</text>',
            f'<text x="{view_x + 6:.3f}" y="{view_y + 40:.3f}" fill="#ffffff" font-size="10">boundary area={signed_area(boundary_open):.3f}</text>',
            f'<text x="{view_x + 6:.3f}" y="{view_y + 52:.3f}" fill="#ffd400" font-size="10">offset area={signed_area(offset):.3f}</text>',
            f'<text x="{view_x + 6:.3f}" y="{view_y + 64:.3f}" fill="#ff4fd8" font-size="10">bevel area={signed_area(bevel):.3f}</text>',
            f'<text x="{view_x + 6:.3f}" y="{view_y + 76:.3f}" fill="#7CFC00" font-size="10">green=start, orange=next</text>',
            draw_poly(zone_poly, "#7dd3fc", "rgba(125,211,252,0.10)", 1.5),
            draw_poly(boundary_open, "#ffffff", "none", 2),
            draw_poly(offset, "#ffd400", "none", 2),
            draw_poly(bevel, "#ff4fd8", "none", 2),
            draw_poly(pack_poly, "#00e5a8", "none", 1),
            draw_markers(boundary_open, "boundary", "#ffffff"),
            draw_markers(offset, "offset", "#ffd400"),
            draw_markers(bevel, "bevel", "#ff4fd8"),
            "</svg>",
        ]
        (OUT_DIR / f"zone_{zid}_bleed_debug.svg").write_text("".join(lines), encoding="utf-8")

        summary_lines.append(
            f"zone {zid}: boundary_area={signed_area(boundary_open):.3f} "
            f"offset_area={signed_area(offset):.3f} bevel_area={signed_area(bevel):.3f}"
        )

    (OUT_DIR / "README.txt").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
