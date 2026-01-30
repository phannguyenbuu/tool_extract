# regions_build.py
import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.geometry import JOIN_STYLE

def cad_to_px_builder(xmin, xmax, ymin, ymax, scale=1.0, margin=20):
    def cad_to_px(x, y):
        px = ((x - xmin) * scale) + margin
        py = ((ymax - y) * scale) + margin
        return px, py
    return cad_to_px

def build_regions(polygons, poly_to_group, bbox, img_src_path, min_area=10.0):
    xmin, ymin, xmax, ymax = bbox
    img_src_bgr = cv2.imread(img_src_path)
    src_h, src_w = img_src_bgr.shape[:2]

    margin = 20
    scale = 1
    width  = int((xmax - xmin) * scale) + 2 * margin
    height = int((ymax - ymin) * scale) + 2 * margin
    width  = max(width, 1000)
    height = max(height, 1000)

    cad_to_px = cad_to_px_builder(xmin, xmax, ymin, ymax, scale, margin)
    img_out = np.full((height, width, 3), 255, dtype=np.uint8)

    regions = []
    i = 0
    for poly_idx, poly in enumerate(polygons):
        if poly.is_empty or poly.area == 0:
            continue
        
        geoms = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        for g in geoms:
            if g.is_empty or g.area == 0:
                continue
                        
            cx, cy = g.centroid.x, g.centroid.y
            u = int(round(cx))
            v = int(round(src_h - 1 - cy))
            if not (0 <= u < src_w and 0 <= v < src_h):
                continue
            b, gclr, r = img_src_bgr[v, u]
            if b > 240 and gclr > 240 and r > 240:
                continue

            i += 1
            coords = list(g.exterior.coords)
            pts = np.array([cad_to_px(x, y) for x, y in coords], dtype=np.float64)

            cx_px, cy_px = cad_to_px(cx, cy)

            group_id = poly_to_group.get(poly_idx, -1)
            regions.append({
                "points": pts.tolist(),
                "pts": pts,
                "label": i,
                "group": group_id,
                "poly_idx": poly_idx,
                "centroid": (float(cx_px), float(cy_px)),
                "color": (int(b), int(gclr), int(r)),
            })
    
    group_ids = sorted({r["group"] for r in regions if r["group"] >= 0})
    group_to_label = {gid: idx + 1 for idx, gid in enumerate(group_ids)}

    for r in regions:
        gid = r.get("group", -1)
        if gid >= 0:
            r["label"] = group_to_label.get(gid, 0)
        else:
            # neu muon, gan 0 hoac giu nguyen
            r["label"] = 0
    
    return regions, img_out, (width, height)
