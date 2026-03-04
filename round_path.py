import re
from pathlib import Path
import numpy as np

svg_path = Path(r"tool_extract/static/outputs/dc59419bb9274669b0011497743ac476_orig.svg")
out_path = svg_path.with_name(svg_path.stem + "_weld.svg")

# regex bắt thuộc tính points="x1,y1 x2,y2 ..."
polyline_re = re.compile(r'(points=")([^"]+)(")')

def parse_points(s: str):
    pts = []
    for token in s.strip().split():
        if "," not in token:
            continue
        x, y = token.split(",")
        pts.append((float(x), float(y)))
    return np.asarray(pts, dtype=np.float64)

def format_points(arr: np.ndarray):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in arr)

def weld_points(pts: np.ndarray, thr_x=0.5, thr_y=0.5) -> np.ndarray:
    """
    weld: mọi điểm có |dx| <= thr_x và |dy| <= thr_y sẽ dùng chung 1 toạ độ (trung bình).
    Gộp trên toàn bộ tập điểm của file để cạnh chung giữa các polyline trùng nhau.
    """
    n = len(pts)
    if n == 0:
        return pts

    used = np.zeros(n, dtype=bool)
    unified = pts.copy()

    for i in range(n):
        if used[i]:
            continue
        xi, yi = pts[i]
        mask = (~used &
                (np.abs(pts[:, 0] - xi) <= thr_x) &
                (np.abs(pts[:, 1] - yi) <= thr_y))
        idx = np.where(mask)[0]
        if len(idx) <= 1:
            used[i] = True
            continue
        mean = pts[idx].mean(axis=0)
        unified[idx] = mean
        used[idx] = True
    return unified

# 1) đọc SVG
text = svg_path.read_text(encoding="utf-8")

# 2) thu tất cả điểm từ mọi polyline
all_pts = []
spans = []  # (start, end) range trong all_pts cho từng polyline

for m in polyline_re.finditer(text):
    pts = parse_points(m.group(2))
    start = len(all_pts)
    all_pts.extend(pts)
    spans.append((m.span(), start, len(all_pts)))

all_pts = np.asarray(all_pts, dtype=np.float64)

# 3) weld toàn bộ
all_pts_weld = weld_points(all_pts, thr_x=10, thr_y=10)

# 4) thay thế lại trong SVG
chunks = []
last_end = 0
for (span, start, end) in spans:
    s, e = span
    # phần trước polyline giữ nguyên
    chunks.append(text[last_end:s])

    # rebuild polyline với points đã weld
    original = text[s:e]
    pts = all_pts_weld[start:end]
    new_points = format_points(pts)
    replaced = polyline_re.sub(
        lambda m: m.group(1) + new_points + m.group(3),
        original,
        count=1
    )
    chunks.append(replaced)
    last_end = e

chunks.append(text[last_end:])
new_text = "".join(chunks)

out_path.write_text(new_text, encoding="utf-8")
print("saved", out_path)
