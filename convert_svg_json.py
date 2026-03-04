import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_points_attr(points_str):
    """
    Parse chuỗi points của <polyline>/<polygon> thành list điểm [{x, y}, ...]
    Hỗ trợ cả 'x1,y1 x2,y2' và 'x1,y1,x2,y2' v.v.
    """
    if not points_str:
        return []
    # Thay , bằng space rồi tách
    raw = points_str.replace(',', ' ').split()
    coords = [float(v) for v in raw]
    if len(coords) % 2 != 0:
        raise ValueError("Số lượng số trong points không chẵn")
    points = []
    for i in range(0, len(coords), 2):
        points.append({"x": coords[i], "y": coords[i+1]})
    return points


def sample_cubic_bezier(p0, p1, p2, p3, steps=20):
    """Lấy mẫu đường cong bezier bậc 3 thành list điểm (x, y)."""
    pts = []
    for i in range(steps + 1):
        t = i / steps
        x = (
            (1 - t) ** 3 * p0[0]
            + 3 * (1 - t) ** 2 * t * p1[0]
            + 3 * (1 - t) * t ** 2 * p2[0]
            + t ** 3 * p3[0]
        )
        y = (
            (1 - t) ** 3 * p0[1]
            + 3 * (1 - t) ** 2 * t * p1[1]
            + 3 * (1 - t) * t ** 2 * p2[1]
            + t ** 3 * p3[1]
        )
        pts.append((x, y))
    return pts


def parse_path_d(d, curve_steps=20):
    import re

    token_re = re.compile(r'([MmLlHhVvZz])|([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)')
    tokens = token_re.findall(d)
    tokens = [t[0] or t[1] for t in tokens]

    i = 0
    cmd = None
    curr = (0.0, 0.0)
    start_subpath = (0.0, 0.0)
    points = []

    def is_cmd(tok):
        return bool(re.match(r'[MmLlHhVvZz]', tok))

    def get_numbers(n):
        nonlocal i
        vals = []
        for _ in range(n):
            if i >= len(tokens):
                raise ValueError("Thiếu số trong d")
            if is_cmd(tokens[i]):
                raise ValueError("Gặp lệnh mới khi đang đọc số")
            vals.append(float(tokens[i]))
            i += 1
        return vals

    while i < len(tokens):
        t = tokens[i]
        if is_cmd(t):
            cmd = t
            i += 1

        if cmd in ('M', 'm'):
            x, y = get_numbers(2)
            if cmd == 'm':
                curr = (curr[0] + x, curr[1] + y)
            else:
                curr = (x, y)
            start_subpath = curr
            points.append({"x": curr[0], "y": curr[1]})
            while i < len(tokens) and not is_cmd(tokens[i]):
                x, y = get_numbers(2)
                if cmd == 'm':
                    curr = (curr[0] + x, curr[1] + y)
                else:
                    curr = (x, y)
                points.append({"x": curr[0], "y": curr[1]})

        elif cmd in ('L', 'l'):
            while i < len(tokens) and not is_cmd(tokens[i]):
                x, y = get_numbers(2)
                if cmd == 'l':
                    curr = (curr[0] + x, curr[1] + y)
                else:
                    curr = (x, y)
                points.append({"x": curr[0], "y": curr[1]})

        elif cmd in ('H', 'h'):
            while i < len(tokens) and not is_cmd(tokens[i]):
                (x,) = get_numbers(1)
                if cmd == 'h':
                    curr = (curr[0] + x, curr[1])
                else:
                    curr = (x, curr[1])
                points.append({"x": curr[0], "y": curr[1]})

        elif cmd in ('V', 'v'):
            while i < len(tokens) and not is_cmd(tokens[i]):
                (y,) = get_numbers(1)
                if cmd == 'v':
                    curr = (curr[0], curr[1] + y)
                else:
                    curr = (curr[0], y)
                points.append({"x": curr[0], "y": curr[1]})

        elif cmd in ('Z', 'z'):
            curr = start_subpath
            points.append({"x": curr[0], "y": curr[1]})

        else:
            raise NotImplementedError(f"Chưa hỗ trợ lệnh path: {cmd}")

    return points

from svg.path import parse_path
from xml.etree import ElementTree as ET
import json

from svg.path import parse_path

def path_to_polyline(d, steps=30):
    path = parse_path(d)          # hiểu đủ M,L,H,V,C,S,Q,T,A,Z
    pts = []
    for i in range(steps + 1):
        t = i / steps             # 0 -> 1
        p = path.point(t)
        pts.append({
            "x": float(p.real),
            "y": float(p.imag),
        })
    return pts



TARGET_W = 2955.0
TARGET_H = 2161.0


def extend_segment(p0, p1, extend_ratio=0.01, min_extend=0.5):
    """
    Kéo dài đoạn thẳng p0->p1 ở cả 2 đầu.
    extend_ratio: tỉ lệ so với độ dài đoạn.
    min_extend: độ dài tối thiểu kéo dài (đơn vị px).
    """
    x0, y0 = p0["x"], p0["y"]
    x1, y1 = p1["x"], p1["y"]

    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return p0, p1

    # độ dài cần extend mỗi đầu
    d = max(length * extend_ratio, min_extend)
    ux = dx / length
    uy = dy / length

    # đầu mới
    ex0 = x0 - ux * d
    ey0 = y0 - uy * d
    ex1 = x1 + ux * d
    ey1 = y1 + uy * d

    return {"x": ex0, "y": ey0}, {"x": ex1, "y": ey1}


def polylines_to_extended_segments(polylines,
                                   extend_ratio=0.01,
                                   min_extend=0.5):
    """
    Biến list polyline thành list các segment đã extend.
    Mỗi segment là polyline 2 điểm.
    """
    segments = []
    for poly in polylines:
        if len(poly) < 2:
            continue
        for i in range(len(poly) - 1):
            p0 = poly[i]
            p1 = poly[i + 1]
            # nếu muốn bỏ các đoạn rất ngắn, có thể check length ở đây
            ep0, ep1 = extend_segment(p0, p1,
                                      extend_ratio=extend_ratio,
                                      min_extend=min_extend)
            segments.append([ep0, ep1])
    return segments


def scale_polylines(polylines, view_w, view_h,
                    target_w=TARGET_W, target_h=TARGET_H,
                    keep_aspect=False,
                    flip_y=True):
    sx = target_w / view_w
    sy = target_h / view_h
    if keep_aspect:
        s = min(sx, sy)
        sx = sy = s

    scaled = []
    for poly in polylines:
        new_poly = []
        for pt in poly:
            x = pt["x"] * sx
            y = pt["y"] * sy
            if flip_y:
                y = target_h - y
            new_poly.append({"x": x, "y": y})
        scaled.append(new_poly)
    return scaled



def svg_to_polylines(svg_path, steps=30):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    vb = root.get("viewBox", "0 0 738.4 540")
    _, _, vw, vh = map(float, vb.split())

    
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'

    polylines = []

    # polygon / polyline giữ nguyên
    from math import isclose

    def parse_points_attr(points_str):
        if not points_str:
            return []
        raw = points_str.replace(',', ' ').split()
        coords = [float(v) for v in raw]
        return [{"x": coords[i], "y": coords[i+1]} for i in range(0, len(coords), 2)]

    for el in root.iter(f'{ns}polygon'):
        pts = parse_points_attr(el.get('points'))
        if pts and not (isclose(pts[0]["x"], pts[-1]["x"]) and isclose(pts[0]["y"], pts[-1]["y"])):
            pts.append(pts[0])
        polylines.append(pts)

    for el in root.iter(f'{ns}polyline'):
        pts = parse_points_attr(el.get('points'))
        if pts:
            polylines.append(pts)

    # path: dùng svg.path
    for el in root.iter(f'{ns}path'):
        d = el.get('d')
        if not d:
            continue
        pts = path_to_polyline(d, steps=steps)
        if pts:
            polylines.append(pts)


    for el in root.iter(f'{ns}rect'):
        x = float(el.get('x', 0))
        y = float(el.get('y', 0))
        w = float(el.get('width', 0))
        h = float(el.get('height', 0))
        pts = [
            {"x": x,     "y": y},
            {"x": x+w,   "y": y},
            {"x": x+w,   "y": y+h},
            {"x": x,     "y": y+h},
            {"x": x,     "y": y},  # đóng lại
        ]
        polylines.append(pts)

    for el in root.iter(f'{ns}line'):
        x1 = float(el.get('x1', 0))
        y1 = float(el.get('y1', 0))
        x2 = float(el.get('x2', 0))
        y2 = float(el.get('y2', 0))
        polylines.append([
            {"x": x1, "y": y1},
            {"x": x2, "y": y2},
        ])

    polylines = scale_polylines(polylines, vw, vh,
                                target_w=2955.0, target_h=2161.0,
                                keep_aspect=False)
    

    return {"polylines": polylines}
    


import json
from pathlib import Path
from PIL import Image, ImageDraw

def draw_polylines(json_path, png_path,
                   bg_color="white",
                   line_color="black",
                   line_width=2):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    polylines = data["polylines"]

    w = int(TARGET_W)
    h = int(TARGET_H)

    img = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(img)

    for poly in polylines:
        pts = [(float(pt["x"]), float(pt["y"])) for pt in poly]
        if len(pts) >= 2:
            draw.line(pts, fill=line_color, width=line_width)

    img.save(png_path)
    print(f"Saved to {png_path}")



if __name__ == "__main__":
    # Đổi đường dẫn này sang file SVG của bạn
    svg_file = "input.svg"
    out_file = "polylines.json"

    data = svg_to_polylines("input.svg", steps=40)
    with open("polylines.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Đã ghi {out_file}")

    draw_polylines("polylines.json", "polylines_preview.png")
