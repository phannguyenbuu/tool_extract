from rectpack import newPacker
import numpy as np
import cv2
from typing import Dict, List, Any
from group_patch import extract_group_patch, rotate_regions_once, find_text_position
from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.validation import make_valid
from shapely.errors import GEOSException
from shapely import remove_repeated_points


MAX_W        = 2955
MAX_H        = 2161
PADDING      = 50
GAP          = 30
BORDER_THICK = 20


# ---------- helpers ----------
def find_label_position_dt(group_mask, bbox):
    """
    Dung Distance Transform de tim 'Pole of Inaccessibility' (iem sau nhat).
    Nhanh hon va chinh xac hon quet luoi vector.
    """
    xmin, ymin, xmax, ymax = bbox
    # Crop mask de xu ly nhanh hon
    x0, y0, x1, y1 = int(xmin), int(ymin), int(xmax), int(ymax)
    h, w = group_mask.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    
    crop = group_mask[y0:y1, x0:x1]
    if crop.size == 0:
        return (xmin + xmax) / 2, (ymin + ymax) / 2

    # Distance Transform tren vung group
    dist_map = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_map)
    
    # Neu iem sau nhat co ban kinh > 2px, dung no
    if max_val > 2.0:
        return float(max_loc[0] + x0), float(max_loc[1] + y0)
    
    # Fallback: Neu ben trong qua hep, tim vung trong gan do nhat (ben ngoai)
    # Nhung hien tai uu tien dung tam bbox cho an toan
    return (xmin + xmax) / 2, (ymin + ymax) / 2


def _build_group_patches(group_to_regions, max_width, max_height):
    group_patches = []
    
    for gid, regs in group_to_regions.items():
        patch, local_regions, (w0, h0) = extract_group_patch(regs)
        best_regions = local_regions
        best_area = w0 * h0
        
        for angle in range(0, 180, 5):
            rot_regions, (rw, rh) = rotate_regions_once(local_regions, angle)
            if rw * rh < best_area:  # Minimize area
                best_area = rw * rh
                best_regions = rot_regions
        
        # GIU FLOAT cho bbox calc
        all_pts = np.concatenate([r["pts"].astype(np.float64) for r in best_regions], axis=0)
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)
        
        # TIGHT INT BBOX (no +1 bias)
        bw = int(np.ceil(x_max - x_min))  # ceil  no truncate
        bh = int(np.ceil(y_max - y_min))
        
        # Normalize TRUOC astype  preserve precision
        for r in best_regions:
            pts = r["pts"].astype(np.float64)  # Ensure 64-bit
            pts[:, 0] -= x_min
            pts[:, 1] -= y_min
            r["pts"] = pts  # Float64 local
            
            cx, cy = r["centroid"]
            r["centroid"] = (cx - x_min, cy - y_min)  # Float centroid
        
        group_patches.append((gid, best_regions, bw, bh))
    
    inner_width = max_width - 2 * PADDING
    scaled = [(gid, regs, orig_w, orig_h) for gid, regs, orig_w, orig_h in group_patches]
    return scaled, inner_width



def _pack_patches(scaled_patches, inner_width, max_height):
    """Rectpack  placements_map, canvas_w, canvas_h."""
    packer = newPacker(rotation=False)
    BIN_W = inner_width
    BIN_H = max_height - 2 * PADDING
    packer.add_bin(BIN_W, BIN_H)

    for gid, regs, sw, sh in scaled_patches:
        packer.add_rect(sw + GAP, sh + GAP, rid=gid)

    packer.pack()

    placements_map = {}
    max_y = 0
    for b, x, y, w, h, rid in packer.rect_list():
        x_off = x + PADDING
        y_off = y + PADDING
        bw = w - GAP
        bh = h - GAP
        placements_map[rid] = (x_off, y_off, bw, bh)
        max_y = max(max_y, y_off + h)

    canvas_h = min(max_y + PADDING, max_height)
    canvas_w = MAX_W
    return placements_map, canvas_w, canvas_h


def build_bleed(group_mask: np.ndarray, canvas_fill: np.ndarray):
    """Giu nguyen nhu file goc  dung canvas_fill e tao bleed-color-img."""
    border = int(BORDER_THICK // 2)
    h, w = group_mask.shape
    bleed_color_img = canvas_fill.copy()

    current_mask = (group_mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(border):
        dilated = cv2.dilate(current_mask, kernel, iterations=1)
        ring = (dilated == 1) & (current_mask == 0)
        if not ring.any():
            break

        src = bleed_color_img.copy()
        yy, xx = np.where(ring)
        for y, x in zip(yy, xx):
            y0, y1 = max(0, y-1), min(h, y+2)
            x0, x1 = max(0, x-1), min(w, x+2)
            neighbours_mask = current_mask[y0:y1, x0:x1]
            neighbours_img = src[y0:y1, x0:x1]
            ys2, xs2 = np.where(neighbours_mask == 1)
            if len(ys2) == 0:
                continue
            bleed_color_img[y, x] = neighbours_img[ys2[0], xs2[0]]

        current_mask = dilated

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*border+1, 2*border+1))
    dilated_final = cv2.dilate(group_mask, kernel2)
    bleed_mask = ((dilated_final > 0) & (group_mask == 0)).astype(np.uint8) * 255

    return bleed_mask, bleed_color_img


# ---------- main API ----------
def build_layout_canvas(
    group_to_regions: Dict[int, List[Dict[str, Any]]],
    max_width: int = MAX_W,
    max_height: int = MAX_H
):
    vector_elements: List[Dict[str, Any]] = []
    scaled_patches, inner_width = _build_group_patches(
        group_to_regions, max_width, max_height
    )
    placements_map, canvas_w, canvas_h = _pack_patches(
        scaled_patches, inner_width, max_height
    )

    canvas_fill = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
    canvas_line = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
    
    # NEW: Raster toan bo truoc e lay Global Occupied Mask
    global_occupied_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    group_masks = {} # Luu lai mask tung group e dung sau

    bleed_contours_by_gid: Dict[int, List[np.ndarray]] = {}
    num_groups = len(scaled_patches)
    print(f"[layout] total groups: {num_groups}")

    # --- PASS 1: Ve toan bo va build Global Mask ---
    for gidx, (gid, regs, sw, sh) in enumerate(scaled_patches):
        if gid not in placements_map: continue
        x_off, y_off, bw, bh = placements_map[gid]
        
        g_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        for r in regs:
            pts = r["pts"].astype(np.float64).copy()
            pts[:, 0] += x_off
            pts[:, 1] += y_off
            pts_i32 = pts.astype(np.int32)
            
            cv2.fillPoly(canvas_fill, [pts_i32], tuple(map(int, r["color"])))
            cv2.fillPoly(g_mask, [pts_i32], 255)
            cv2.fillPoly(global_occupied_mask, [pts_i32], 255)
            
            # Save vector polygons
            vector_elements.append({
                "type": "polygon",
                "gid": gid,
                "pts": [[float(x), float(y)] for x, y in pts],
                "color": [float(c) for c in r["color"]],
            })
        group_masks[gid] = g_mask

    # Distance Transform tren Global Background (optional, co the dung e tim vung trong)
    # global_bg_dt = cv2.distanceTransform(255 - global_occupied_mask, cv2.DIST_L2, 5)

    # --- PASS 2: Outlines, Bleeds va Labels ---
    for gidx, (gid, regs, sw, sh) in enumerate(scaled_patches):
        if gid not in placements_map: continue
        x_off, y_off, bw, bh = placements_map[gid]
        group_mask = group_masks[gid]

        # BUILD BLEED
        bleed_mask, bleed_color_img = build_bleed(group_mask, canvas_fill)
        canvas_fill[:] = bleed_color_img
        contours, _ = cv2.findContours(bleed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bleed_contours_by_gid[gid] = [cnt.reshape(-1, 2) for cnt in contours]

        # OUTLINE + LABEL
        polys = []
        for r in regs:
            pts = r["pts"].astype(np.float64).copy()
            pts[:, 0] += x_off
            pts[:, 1] += y_off
            if len(pts) >= 3: polys.append(Polygon(pts))

        if polys:
            try:
                valid_polys = [make_valid(p) for p in polys if p.is_valid or not make_valid(p).is_empty]
                if valid_polys:
                    merged = unary_union(valid_polys)
                    if not merged.is_valid: merged = make_valid(merged)
                    
                    # Simplify & Outline (giu logic cu)
                    merged = merged.simplify(0.8, preserve_topology=True)
                    geoms = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)
                    for g_poly in geoms:
                        ring = remove_repeated_points(g_poly.exterior, tolerance=1)
                        vector_elements.append({
                            "type": "outline",
                            "gid": gid,
                            "coords": [[float(x), float(y)] for x, y in ring.coords],
                        })

                    # LABEL logic moi: Distance Transform
                    label_poly = merged if merged.geom_type == "Polygon" else max(merged.geoms, key=lambda g: g.area)
                    group_label = str(regs[0]["label"]) # Lay label tu reg au tien
                    
                    text_x, text_y = find_label_position_dt(group_mask, label_poly.bounds)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fs = 0.4
                    (tw, th), _ = cv2.getTextSize(group_label, font, fs, 1)
                    org_x, org_y = int(text_x - tw / 2), int(text_y + th / 2)
                    cv2.putText(canvas_line, group_label, (org_x, org_y), font, fs, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    vector_elements.append({
                        'type': 'label', 'gid': gid, 'text': group_label,
                        'x': text_x, 'y': text_y
                    })
            except Exception as e:
                print(f"[layout] Error gid={gid}: {e}")

    return canvas_fill, canvas_line, vector_elements, bleed_contours_by_gid

    return canvas_fill, canvas_line, vector_elements, bleed_contours_by_gid
