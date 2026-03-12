# polygonize_utils.py
import json
from math import hypot
from shapely.geometry import LineString, GeometryCollection
from shapely.ops import unary_union, polygonize_full

def load_polylines(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = [
        [(pt["x"], pt["y"]) for pt in poly]
        for poly in data["polylines"]
    ]
    if not lines:
        raise SystemExit("Khong co polyline nao trong file.")
    return lines

SNAP_GRID = 0.1  # luoi 0.1 on vi

def snap_point(x, y, s=SNAP_GRID):
    return round(x / s) * s, round(y / s) * s

def build_segments(lines, snap=0.2):
    segments = []
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for poly in lines:
        if len(poly) < 2:
            continue
        for (x1, y1), (x2, y2) in zip(poly, poly[1:]):
            x1s, y1s = snap_point(x1, y1, snap)
            x2s, y2s = snap_point(x2, y2, snap)
            if (x1s, y1s) == (x2s, y2s):
                continue
            segments.append(LineString([(x1s, y1s), (x2s, y2s)]))
            xmin = min(xmin, x1s, x2s)
            ymin = min(ymin, y1s, y2s)
            xmax = max(xmax, x1s, x2s)
            ymax = max(ymax, y1s, y2s)
    if not segments:
        raise SystemExit("Khong tao uoc segment nao.")
    return segments, (xmin, ymin, xmax, ymax)

def auto_join_gaps(segments, join_dist=2.0):
    endpoints = []
    for seg in segments:
        x1, y1 = seg.coords[0]
        x2, y2 = seg.coords[-1]
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))
    extra_segments = []
    n = len(endpoints)
    for i in range(n):
        x1, y1 = endpoints[i]
        for j in range(i + 1, n):
            x2, y2 = endpoints[j]
            d = hypot(x2 - x1, y2 - y1)
            if 0 < d <= join_dist:
                extra_segments.append(LineString([(x1, y1), (x2, y2)]))
    segments.extend(extra_segments)
    return segments, len(extra_segments)

def polygonize_segments(segments):
    merged = unary_union(segments)
    polys, dangles, cuts, invalids = polygonize_full(merged)
    if isinstance(polys, GeometryCollection):
        polygons = list(polys.geoms)
    else:
        polygons = [polys]
    if not polygons:
        raise SystemExit("Khong polygonize uoc region.")
    return polygons
