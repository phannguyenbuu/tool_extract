from collections import defaultdict
import base64
from typing import Dict, List, Any

import cv2
import numpy as np
from shapely.geometry import Polygon, Point, LineString, LinearRing
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely import remove_repeated_points

def find_text_position(poly: Polygon, bbox: tuple) -> tuple[float, float]:
    """
    Lay uong ngang qua bbox.y/2, cat polygon, 
    roi lay trung iem cua 2 iem giao au tien.
    """
    xmin, ymin, xmax, ymax = bbox

    # 1) centroid, neu nam trong thi uu tien dung luon
    c = poly.centroid
    if poly.contains(c):
        return float(c.x), float(c.y)

    # 2) uong ngang qua giua bbox
    mid_y = (ymin + ymax) / 2.0
    h_line = LineString([(xmin, mid_y), (xmax, mid_y)])
    inter = poly.intersection(h_line)

    xs = []

    if inter.is_empty:
        pass
    elif inter.geom_type == "Point":
        xs = [inter.x]
    elif inter.geom_type == "MultiPoint":
        xs = [p.x for p in inter.geoms]
    elif inter.geom_type == "LineString":
        # cat trung theo oan: lay 2 au mut
        xs = [inter.coords[0][0], inter.coords[-1][0]]
    elif inter.geom_type == "MultiLineString":
        for ln in inter.geoms:
            xs.extend([ln.coords[0][0], ln.coords[-1][0]])

    xs = sorted(xs)

    # can it nhat 2 giao iem e lay trung iem
    if len(xs) >= 2:
        x1, x2 = xs[0], xs[1]
        mx = 0.5 * (x1 + x2)
        pt = Point(mx, mid_y)
        # neu vi rounding ma ra ngoai thi van chap nhan; 
        # con muon chac chan thi kiem tra:
        if poly.contains(pt) or poly.touches(pt):
            return float(mx), float(mid_y)

    # 3) fallback: centroid bbox
    bx = 0.5 * (xmin + xmax)
    by = 0.5 * (ymin + ymax)
    return float(bx), float(by)



def snap_axis_aligned(ring: LinearRing, tol: float = 3.0) -> LinearRing:
    """
    Neu oan gan nhu thang ung / ngang (|dx| hoac |dy| < tol),
    ep cac inh cua oan o ve cung x hoac cung y.
    tol: on vi pixel tren canvas.
    """
    coords = np.asarray(ring.coords, dtype=np.float64)
    if len(coords) <= 4:
        return ring

    base = coords[:-1].copy()
    n = len(base)

    for i in range(n):
        j = (i + 1) % n
        x1, y1 = base[i]
        x2, y2 = base[j]

        dx = x2 - x1
        dy = y2 - y1

        # gan vertical
        if abs(dx) < tol and abs(dy) > tol:
            x_snap = 0.5 * (x1 + x2)
            base[i, 0] = x_snap
            base[j, 0] = x_snap

        # gan horizontal
        elif abs(dy) < tol and abs(dx) > tol:
            y_snap = 0.5 * (y1 + y2)
            base[i, 1] = y_snap
            base[j, 1] = y_snap

    snapped = np.vstack([base, base[0]])
    return LinearRing(snapped)



def normalize_group_polys(
    polys: List[Polygon],
    simplify_eps: float = 0.5,
    min_seg_len: float = 0.5
) -> List[Polygon]:
    """
    Tach rieng logic UNION tu build_layout_canvas  reusable
    
    Returns: list clean Polygon(s) sau union + simplify + remove_repeated_points
    """
    if not polys:
        return []
    
    try:
        # 1. Validate tung polygon
        valid_polys = []
        for poly in polys:
            if poly.is_valid and not poly.is_empty:
                valid_polys.append(poly)
            else:
                fixed_poly = make_valid(poly)
                if not fixed_poly.is_empty:
                    valid_polys.append(fixed_poly)
        
        if not valid_polys:
            return []
        
        # 2. Safe unary_union
        merged = unary_union(valid_polys)
        if merged.is_empty:
            return []
        
        # 3. am bao valid
        if not merged.is_valid:
            merged = make_valid(merged)
        if merged.is_empty:
            return []
        
        # 4. Simplify
        if simplify_eps > 0:
            try:
                merged = merged.simplify(simplify_eps, preserve_topology=True)
            except:
                pass  # keep original
        
        # 5. Xu ly MultiPolygon  list geoms
        if merged.geom_type == "Polygon":
            geoms = [merged]
        elif merged.geom_type == "MultiPolygon":
            geoms = list(merged.geoms)
        else:
            return []
        
        # 6. Clean repeated points cho tung geom
        clean_geoms = []
        for g_poly in geoms:
            if g_poly.is_empty or not g_poly.is_valid:
                continue
            
            try:
                ring = remove_repeated_points(g_poly.exterior, tolerance=min_seg_len)
                g_clean = Polygon(ring)
                if not g_clean.is_empty:
                    clean_geoms.append(g_clean)
            except:
                continue
        
        return clean_geoms
        
    except Exception as e:
        print(f"[normalize_group_polys] Error: {e}")
        return []

def export_orig_svg(regions, width: int, height: int, outpath: str):
    grouptopolys = defaultdict(list)
    grouptolabel = {}
    
    # Gom poly (giu nguyen)
    for r in regions:
        gid = r.get("group", -1)
        if gid < 0: continue
        pts = np.array(r["points"], dtype=np.float64)
        if len(pts) < 3: continue
        poly = Polygon(pts)
        if poly.is_empty: continue
        grouptopolys[gid].append(poly)
        lab = r.get("label", 0)
        if lab > 0: grouptolabel[gid] = lab
    
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    
    fallback_labels = {}
    if grouptopolys and not grouptolabel:
        ordered = sorted(grouptopolys.keys())
        fallback_labels = {g: i + 1 for i, g in enumerate(ordered)}

    for gid, polys in grouptopolys.items():
        if not polys: continue
        
        merged = unary_union(polys)  # Union OK
        if merged.is_empty:
            continue

        if merged.geom_type == "Polygon":
            geoms = [merged]
        else:
            geoms = list(merged.geoms)

        for geom in geoms:
            if geom.is_empty or not hasattr(geom, "exterior"):
                continue
            coords = np.array(geom.exterior.coords, dtype=np.float64)
            pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
            lines.append(
                f'<polyline points="{pts_str}" fill="none" stroke="#000000" '
                f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        
        label = str(grouptolabel.get(gid, fallback_labels.get(gid, gid + 1)))

        label_poly = max(geoms, key=lambda g: g.area)
        bbox = label_poly.bounds  # (xmin,ymin,xmax,ymax)
        tx, ty = find_text_position(label_poly, bbox)
        
        lines.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-family="Arial,Helvetica,sans-serif" font-size="16" font-weight="bold" fill="#ff0000" text-anchor="middle" dominant-baseline="middle">{label}</text>')
    
    lines.append("</svg>")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("".join(lines))



def save_b64_png(b64_data: str, output_path: str):
    # Decode base64  bytes
    png_bytes = base64.b64decode(b64_data)
    
    # Bytes  numpy array
    nparr = np.frombuffer(png_bytes, np.uint8)
    
    # Decode  image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Save
    cv2.imwrite(output_path, img)
    # print(f" Saved PNG: {output_path} ({png_mb:.1f}MB)")


def export_hybrid_svg(
    canvas_fill: np.ndarray,
    vector_elements: List[Dict[str, Any]],
    bleed_contours_by_gid: Dict[int, List[np.ndarray]],
    width: int,
    height: int,
    out_path: str,
    scale_factor: int = 2,
):
    # 1) Upscale raster
    w2, h2 = int(width * scale_factor), int(height * scale_factor)
    hi_res = cv2.resize(canvas_fill, (w2, h2), interpolation=cv2.INTER_CUBIC)

    ok, buf = cv2.imencode(".png", hi_res, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("imencode PNG failed")

    png_mb = len(buf) / (1024 * 1024)
    b64 = base64.b64encode(buf).decode("utf-8")

    save_b64_png(b64, out_path.replace(".svg",".png"))
    print(f"[hybrid_svg] raster {w2}x{h2}, PNG ~{png_mb:.1f} MB")
    
    # 3) SVG
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]

    # 4) Raster layer
    inv_scale = 1.0 / scale_factor
    lines.append(
        f'''
    <!-- Raster bleed 2x  scale {inv_scale:.3f} -->
    <g transform="scale({inv_scale})">
        <image href="data:image/png;base64,{b64}"
               x="0" y="0"
               width="{w2}" height="{h2}"
               preserveAspectRatio="none"
               image-rendering="optimizeQuality"/>
    </g>'''
    )

    # 5a) Outline vector (dung coords a normalize)
    for el in vector_elements:
        if not isinstance(el, dict):
            continue
        if el.get("type", "") != "outline":
            continue

        coords = el.get("coords", [])
        if not (isinstance(coords, list) and len(coords) > 2):
            continue

        pts_str = " ".join(f"{int(x)},{int(y)}" for x, y in coords)
        lines.append(
            f'<polyline points="{pts_str}" '
            f'fill="none" stroke="#000000" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    total_pieces = 0
    # 5b) Label vector
    for el in vector_elements:
        if not isinstance(el, dict):
            continue
        if el.get("type", "") != "label":
            continue

        x = float(el.get("x", 0))
        y = float(el.get("y", 0))
        text = str(el.get("text", "")).strip()
        if not text:
            continue

        lines.append(
            f'<text x="{x}" y="{y}" '
            f'font-family="Arial,Helvetica,sans-serif" '
            f'font-size="14" font-weight="bold" fill="#000000" '
            f'text-anchor="middle" dominant-baseline="middle">{text}</text>'
        )

        total_pieces += 1








    lines.insert(3, f'  <g transform="translate(25, 40) scale(1)">')  # Corner safe
    lines.insert(4, f'    <text x="0" y="22" font-family="Arial,Helvetica,sans-serif" ')
    lines.insert(5, f'            font-size="32" font-weight="bold"')
    lines.insert(6, f'            text-anchor="start" dominant-baseline="middle">Pieces: {total_pieces}</text>')
    lines.insert(7, f'  </g>')






    

    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[hybrid_svg] saved {out_path}")


from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import Dict, List, Any
import numpy as np

def export_line_svg(
    vector_elements: List[Dict[str, Any]],
    width: int,
    height: int,
    out_path: str,
):
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]

    # outline
    for el in vector_elements:
        if not isinstance(el, dict):
            continue
        if el.get("type", "") != "outline":
            continue

        coords = el.get("coords", [])
        if not (isinstance(coords, list) and len(coords) > 2):
            continue

        pts_str = " ".join(f"{int(x)},{int(y)}" for x, y in coords)
        lines.append(
            f'<polyline points="{pts_str}" '
            f'fill="none" stroke="#000000" stroke-width="4" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    # label
    for el in vector_elements:
        if not isinstance(el, dict):
            continue
        if el.get("type", "") != "label":
            continue

        x = float(el.get("x", 0))
        y = float(el.get("y", 0))
        text = str(el.get("text", "")).strip()
        if not text:
            continue

        lines.append(
            f'<text x="{x}" y="{y}" '
            f'font-family="Arial,Helvetica,sans-serif" '
            f'font-size="14" font-weight="bold" fill="#000000" '
            f'text-anchor="middle" dominant-baseline="middle">{text}</text>'
        )

    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
