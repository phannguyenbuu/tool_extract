from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from math import ceil
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.parse import unquote

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from . import config


def _parse_points(points_str: str) -> List[Tuple[float, float]]:
    if not points_str:
        return []
    tokens: List[str] = []
    for part in points_str.replace("\n", " ").replace("\t", " ").split():
        if "," in part:
            tokens.extend(part.split(","))
        else:
            tokens.append(part)
    pts: List[Tuple[float, float]] = []
    if len(tokens) % 2 == 0:
        for i in range(0, len(tokens), 2):
            try:
                x = float(tokens[i])
                y = float(tokens[i + 1])
                pts.append((x, y))
            except ValueError:
                continue
        return pts
    for part in points_str.strip().split():
        if "," not in part:
            continue
        x_str, y_str = part.split(",", 1)
        try:
            pts.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    return pts


def _iter_geometry(root: ET.Element) -> Iterable[Tuple[str, List[Tuple[float, float]]]]:
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "polyline":
            pts = _parse_points(el.attrib.get("points", ""))
            if len(pts) >= 2:
                yield ("polyline", pts)
        elif tag == "polygon":
            pts = _parse_points(el.attrib.get("points", ""))
            if len(pts) >= 3:
                yield ("polygon", pts)
        elif tag == "line":
            try:
                x1 = float(el.attrib.get("x1", "0"))
                y1 = float(el.attrib.get("y1", "0"))
                x2 = float(el.attrib.get("x2", "0"))
                y2 = float(el.attrib.get("y2", "0"))
                yield ("polyline", [(x1, y1), (x2, y2)])
            except ValueError:
                continue


def _get_canvas_size(root: ET.Element, scale: float) -> Tuple[int, int]:
    vb = root.attrib.get("viewBox")
    if vb:
        parts = vb.replace(",", " ").split()
        if len(parts) == 4:
            _, _, w, h = parts
            return (int(ceil(float(w) * scale)), int(ceil(float(h) * scale)))
    w_attr = root.attrib.get("width")
    h_attr = root.attrib.get("height")
    if w_attr and h_attr:
        try:
            return (int(ceil(float(w_attr) * scale)), int(ceil(float(h_attr) * scale)))
        except ValueError:
            pass
    raise RuntimeError("Cannot determine SVG canvas size")


def _parse_transform(transform: str) -> List[Tuple[str, Tuple[float, float]]]:
    ops: List[Tuple[str, Tuple[float, float]]] = []
    if not transform:
        return ops
    for name, args in re.findall(r"(translate|scale)\s*\(([^)]*)\)", transform):
        nums = [float(v) for v in re.split(r"[ ,]+", args.strip()) if v]
        if name == "translate":
            tx = nums[0] if len(nums) > 0 else 0.0
            ty = nums[1] if len(nums) > 1 else 0.0
            ops.append(("translate", (tx, ty)))
        elif name == "scale":
            sx = nums[0] if len(nums) > 0 else 1.0
            sy = nums[1] if len(nums) > 1 else sx
            ops.append(("scale", (sx, sy)))
    return ops


def _ops_to_matrix(ops: List[Tuple[str, Tuple[float, float]]]) -> np.ndarray:
    m = np.eye(3, dtype=np.float64)
    for name, params in ops:
        if name == "translate":
            tx, ty = params
            t = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
            m = t @ m
        elif name == "scale":
            sx, sy = params
            s = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
            m = s @ m
    return m


def _invert_transform_point(x: float, y: float, m: np.ndarray) -> Tuple[float, float]:
    inv = np.linalg.inv(m)
    v = np.array([x, y, 1.0], dtype=np.float64)
    out = inv @ v
    return float(out[0]), float(out[1])


def _normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


def _find_embedded_image(root: ET.Element) -> Tuple[Path, np.ndarray, Tuple[float, float]]:
    href_key = "{http://www.w3.org/1999/xlink}href"
    svg_dir = config.SVG_PATH.parent
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag != "image":
            continue
        href = el.attrib.get(href_key) or el.attrib.get("href")
        if not href:
            continue
        href = unquote(href)
        img_path = Path(href)
        if not img_path.is_absolute():
            img_path = (svg_dir / img_path).resolve()
        transform = el.attrib.get("transform", "")
        ops = []
        x_attr = float(el.attrib.get("x", "0"))
        y_attr = float(el.attrib.get("y", "0"))
        if x_attr != 0.0 or y_attr != 0.0:
            ops.append(("translate", (x_attr, y_attr)))
        ops.extend(_parse_transform(transform))
        if len(ops) == 2 and ops[0][0] == "translate" and ops[1][0] == "scale":
            tx, ty = ops[0][1]
            sx, sy = ops[1][1]
            m = np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=np.float64)
        else:
            m = _ops_to_matrix(ops)
        w = float(el.attrib.get("width", "0"))
        h = float(el.attrib.get("height", "0"))
        if img_path.exists():
            return img_path, m, (w, h)

        basename = img_path.name
        target = _normalize_name(basename)
        candidates: List[Path] = []
        for p in svg_dir.rglob("*"):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
                continue
            if _normalize_name(p.name) == target:
                candidates.append(p)
        if candidates:
            return candidates[0], m, (w, h)
    raise RuntimeError("No embedded <image> found in SVG")


def _read_image_any(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im)
        return arr[:, :, ::-1].copy()


def _write_svg_paths(path: Path, width: int, height: int, polys: List[List[Tuple[float, float]]]) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="none" stroke="#000" stroke-width="1"/>')
    for pts in polys:
        if len(pts) < 2:
            continue
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in pts) + " Z"
        parts.append(f'<path d="{d}" fill="none" stroke="#000" stroke-width="1"/>')
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def _write_svg_paths_fill(path: Path, width: int, height: int, polys: List[List[Tuple[float, float]]], color: str) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="none" stroke="#000" stroke-width="1"/>')
    for pts in polys:
        if len(pts) < 2:
            continue
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in pts) + " Z"
        parts.append(f'<path d="{d}" fill="{color}" stroke="none"/>')
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def _write_svg_paths_fill_stroke(
    path: Path,
    width: int,
    height: int,
    polys: List[List[Tuple[float, float]]],
    fill: str,
    stroke: str,
    opacity: float = 0.6,
) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#0b0f1f"/>')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="none" stroke="#000" stroke-width="1"/>')
    for pts in polys:
        if len(pts) < 2:
            continue
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in pts) + " Z"
        parts.append(
            f'<path d="{d}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="0.5"/>'
        )
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def write_region_svg(polys: List[List[Tuple[float, float]]], canvas: Tuple[int, int]) -> None:
    w, h = canvas
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="none" stroke="#000" stroke-width="1"/>')
    for rid, pts in enumerate(polys):
        if len(pts) < 2:
            continue
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in pts) + " Z"
        parts.append(f'<path d="{d}" fill="#bbb" stroke="#000" stroke-width="1"/>')
        poly = Polygon(pts)
        if poly.is_empty:
            continue
        cx, cy = poly.centroid.x, poly.centroid.y
        parts.append(
            f'<text x="{cx}" y="{cy}" font-size="6" text-anchor="middle" '
            f'dominant-baseline="middle" fill="#111">{rid}</text>'
        )
    parts.append("</svg>")
    config.OUT_REGION_SVG.write_text("".join(parts), encoding="utf-8")


def write_zone_svg(
    polys: List[List[Tuple[float, float]]],
    zone_boundaries: dict,
    canvas: Tuple[int, int],
    colors: List[Tuple[int, int, int]],
) -> None:
    w, h = canvas
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="none" stroke="#000" stroke-width="1"/>')
    parts.append('<g id="fill">')
    for rid, pts in enumerate(polys):
        if len(pts) < 2:
            continue
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in pts) + " Z"
        b, g, r = colors[rid]
        parts.append(f'<path d="{d}" fill="rgb({r},{g},{b})" stroke="none"/>')
    parts.append("</g>")
    parts.append('<g id="outline">')
    for _, lines in zone_boundaries.items():
        for pts in lines:
            if len(pts) < 2:
                continue
            d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in pts)
            parts.append(f'<path d="{d}" fill="none" stroke="#000" stroke-width="1"/>')
    parts.append("</g></svg>")
    config.OUT_ZONE_SVG.write_text("".join(parts), encoding="utf-8")


def write_zone_outline_svg(zone_boundaries: dict, canvas: Tuple[int, int]) -> None:
    paths: List[List[Tuple[float, float]]] = []
    for _, lines in zone_boundaries.items():
        for pts in lines:
            if len(pts) >= 2:
                paths.append(pts)
    _write_svg_paths(config.OUT_ZONE_OUTLINE_SVG, canvas[0], canvas[1], paths)
