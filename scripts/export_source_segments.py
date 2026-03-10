from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from . import source_voronoi
from . import svg_utils


VERTEX_SNAP = 1e-4
VERTEX_MERGE_DIST = 0.75


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1].lower()


def _parse_viewbox(root: ET.Element) -> str:
    vb = root.get("viewBox")
    if vb:
        return vb
    w = root.get("width", "1000").replace("px", "")
    h = root.get("height", "1000").replace("px", "")
    return f"0 0 {w} {h}"


def collect_segments(svg_path: Path) -> tuple[list[list[list[float]]], str]:
    root = ET.parse(svg_path).getroot()
    view_box = _parse_viewbox(root)
    segments: list[list[list[float]]] = []

    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag in {"defs", "clippath"}:
            continue
        if tag == "line":
            try:
                x1 = float(elem.attrib.get("x1", "0"))
                y1 = float(elem.attrib.get("y1", "0"))
                x2 = float(elem.attrib.get("x2", "0"))
                y2 = float(elem.attrib.get("y2", "0"))
            except ValueError:
                continue
            if (x1, y1) != (x2, y2):
                segments.append([[x1, y1], [x2, y2]])
        elif tag == "polyline":
            pts = svg_utils._parse_points(elem.attrib.get("points", ""))
            for a, b in zip(pts, pts[1:]):
                if a != b:
                    segments.append([[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]])
        elif tag == "polygon":
            pts = svg_utils._parse_points(elem.attrib.get("points", ""))
            if len(pts) >= 3:
                for a, b in zip(pts, pts[1:] + [pts[0]]):
                    if a != b:
                        segments.append([[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]])
        elif tag == "path":
            d = elem.attrib.get("d", "")
            for pts in source_voronoi._parse_path_d(d):
                if len(pts) >= 2:
                    for a, b in zip(pts, pts[1:]):
                        if a != b:
                            segments.append([[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]])

    return segments, view_box


def write_segments_svg(out_path: Path, segments: list[list[list[float]]], view_box: str) -> None:
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">']
    for seg in segments:
        a, b = seg
        parts.append(
            f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" '
            'stroke="#000000" stroke-width="0.5" fill="none"/>'
        )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _vertex_key(pt: list[float] | tuple[float, float]) -> tuple[float, float]:
    x, y = float(pt[0]), float(pt[1])
    return (round(x / VERTEX_SNAP) * VERTEX_SNAP, round(y / VERTEX_SNAP) * VERTEX_SNAP)


def _merge_vertices(points: list[list[float]], max_dist: float) -> tuple[list[list[float]], dict[tuple[float, float], int]]:
    clusters: list[dict[str, float | int]] = []
    point_to_vid: dict[tuple[float, float], int] = {}

    for pt in points:
        key = _vertex_key(pt)
        if key in point_to_vid:
            continue
        x, y = key
        match_idx = None
        best_dist = None
        for idx, cluster in enumerate(clusters):
            cx = float(cluster["x"])
            cy = float(cluster["y"])
            dist = math.hypot(x - cx, y - cy)
            if dist <= max_dist and (best_dist is None or dist < best_dist):
                best_dist = dist
                match_idx = idx
        if match_idx is None:
            clusters.append({"x": x, "y": y, "count": 1})
            point_to_vid[key] = len(clusters) - 1
            continue

        cluster = clusters[match_idx]
        count = int(cluster["count"])
        cx = float(cluster["x"])
        cy = float(cluster["y"])
        new_count = count + 1
        cluster["x"] = (cx * count + x) / new_count
        cluster["y"] = (cy * count + y) / new_count
        cluster["count"] = new_count
        point_to_vid[key] = match_idx

    vertices = [[float(cluster["x"]), float(cluster["y"])] for cluster in clusters]
    return vertices, point_to_vid


def dedup_segments_by_adjacency(
    segments: list[list[list[float]]],
) -> tuple[list[list[float]], list[list[int]], list[list[list[float]]]]:
    all_points: list[list[float]] = []
    for seg in segments:
        all_points.extend(seg)
    vertices, vertex_index = _merge_vertices(all_points, VERTEX_MERGE_DIST)
    dedup_keys: set[tuple[int, int]] = set()
    dedup_segments_idx: list[list[int]] = []
    dedup_segments_pts: list[list[list[float]]] = []

    def get_vid(pt: list[float]) -> int:
        key = _vertex_key(pt)
        return int(vertex_index[key])

    for seg in segments:
        a, b = seg
        va = get_vid(a)
        vb = get_vid(b)
        if va == vb:
            continue
        key = (va, vb) if va < vb else (vb, va)
        if key in dedup_keys:
            continue
        dedup_keys.add(key)
        dedup_segments_idx.append([key[0], key[1]])
        dedup_segments_pts.append([vertices[key[0]], vertices[key[1]]])

    return vertices, dedup_segments_idx, dedup_segments_pts


def export_source_segments(svg_path: Path) -> tuple[Path, Path, Path, Path, int, int]:
    segments, view_box = collect_segments(svg_path)
    vertices, dedup_segments_idx, dedup_segments_pts = dedup_segments_by_adjacency(segments)
    stem = svg_path.stem
    svg_out = svg_path.with_name(f"{stem}_segments.svg")
    json_out = svg_path.with_name(f"{stem}_segments.json")
    dedup_svg_out = svg_path.with_name(f"{stem}_segments_dedup.svg")
    dedup_json_out = svg_path.with_name(f"{stem}_segments_dedup.json")
    write_segments_svg(svg_out, segments, view_box)
    write_segments_svg(dedup_svg_out, dedup_segments_pts, view_box)
    json_out.write_text(
        json.dumps({"source": svg_path.name, "segment_count": len(segments), "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dedup_json_out.write_text(
        json.dumps(
            {
                "source": svg_path.name,
                "vertex_snap": VERTEX_SNAP,
                "vertex_merge_dist": VERTEX_MERGE_DIST,
                "vertex_count": len(vertices),
                "segment_count_raw": len(segments),
                "segment_count_dedup": len(dedup_segments_idx),
                "vertices": vertices,
                "segments_idx": dedup_segments_idx,
                "segments": dedup_segments_pts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return svg_out, json_out, dedup_svg_out, dedup_json_out, len(segments), len(dedup_segments_idx)


if __name__ == "__main__":
    source = Path(__file__).resolve().parents[1] / "sources" / "chobenthanh.svg"
    svg_out, json_out, dedup_svg_out, dedup_json_out, count_raw, count_dedup = export_source_segments(source)
    print(f"segments_raw={count_raw}")
    print(f"segments_dedup={count_dedup}")
    print(svg_out)
    print(json_out)
    print(dedup_svg_out)
    print(dedup_json_out)
