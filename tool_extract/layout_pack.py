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
    bleed_contours_by_gid: Dict[int, List[np.ndarray]] = {}
    num_groups = len(scaled_patches)
    print(f"[layout] total groups: {num_groups}")

    for gidx, (gid, regs, sw, sh) in enumerate(scaled_patches):
        if gid not in placements_map:
            continue

        print(f"[layout] group {gidx+1}/{num_groups} (gid={gid})")
        x_off, y_off, bw, bh = placements_map[gid]
        polys = []

        # PASS 1: blend multiply len canvas_fill
        for r in regs:
            pts = r["pts"].copy()
            pts[:, 0] += x_off
            pts[:, 1] += y_off
            pts = pts.astype(np.float64)

            if len(pts) >= 3:
                polys.append(Polygon(pts))

        # PASS 2: fill phang + mask group
        canvas_h, canvas_w = canvas_fill.shape[:2]
        group_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        group_label = None

        for r in regs:
            pts = r["pts"].astype(np.float64).copy()
            pts[:, 0] += x_off
            pts[:, 1] += y_off
            color = tuple(map(int, r["color"]))

            pts_int32 = np.array(pts, dtype=np.int32)

            # Ve
            # print(color)
            cv2.fillPoly(canvas_fill, [pts_int32], color)
            cv2.fillPoly(group_mask, [pts_int32], 255)
            
            # polygon vector data
            pts_vec = r["pts"].astype(np.float32).copy()
            pts_vec[:, 0] += x_off
            pts_vec[:, 1] += y_off
            pts_list = [[float(x), float(y)] for x, y in pts_vec]
            color_list = [float(c) for c in r["color"]]

            vector_elements.append({
                "type": "polygon",
                "gid": gid,
                "pts": pts_list,
                "color": color_list,
            })

            if group_label is None:
                group_label = str(r["label"])

        # BUILD BLEED tren cung canvas_fill
        bleed_mask, bleed_color_img = build_bleed(group_mask, canvas_fill)
        canvas_fill[:] = bleed_color_img

        contours, _ = cv2.findContours(
            bleed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bleed_contours = [cnt.reshape(-1, 2) for cnt in contours]
        bleed_contours_by_gid[gid] = bleed_contours

        # OUTLINE + LABEL  canvas_line + vector_elements (FIXED VERSION)
        SMOOTH_EPS = 1.5
        SIMPLIFY_EPS = 0.8
        MIN_SEG_LEN = 1

        if polys:
            try:
                # 1. Validate tung polygon truoc khi union
                valid_polys = []
                for poly in polys:
                    if poly.is_valid and not poly.is_empty:
                        valid_polys.append(poly)
                    else:
                        # Fix invalid polygons
                        fixed_poly = make_valid(poly)
                        if not fixed_poly.is_empty:
                            valid_polys.append(fixed_poly)

                if valid_polys:
                    # 2. Safe unary_union voi error handling
                    merged = unary_union(valid_polys)
                    
                    if merged.is_empty:
                        print(f"[layout] Empty merged geometry for gid={gid}")
                        continue
                    
                    # 3. am bao valid sau union
                    if not merged.is_valid:
                        merged = make_valid(merged)
                    
                    if merged.is_empty:
                        print(f"[layout] Empty after make_valid for gid={gid}")
                        continue

                    # 4. Smooth neu can (optional - uncomment neu muon)
                    # merged = merged.buffer(SMOOTH_EPS).buffer(-SMOOTH_EPS)

                    # 5. Simplify an toan
                    if SIMPLIFY_EPS > 0:
                        try:
                            merged = merged.simplify(SIMPLIFY_EPS, preserve_topology=True)
                        except GEOSException:
                            print(f"[layout] Simplify failed for gid={gid}, skipping")

                    # 6. Xu ly MultiPolygon
                    if merged.geom_type == "Polygon":
                        geoms = [merged]
                    else:
                        geoms = list(merged.geoms)

                    for g_poly in geoms:
                        if g_poly.is_empty or not g_poly.is_valid:
                            continue

                        try:
                            # Clean repeated points
                            ring = remove_repeated_points(
                                g_poly.exterior, tolerance=MIN_SEG_LEN
                            )
                            
                            coords = np.array(ring.coords, dtype=np.float32)

                            # cv2.polylines(
                            #     canvas_line, [coords_int], True, (0, 0, 0), 2, cv2.LINE_AA
                            # )

                            coords_list = [[float(x), float(y)] for x, y in coords]
                            vector_elements.append({
                                "type": "outline",
                                "gid": gid,
                                "coords": coords_list,
                            })
                        except Exception as e:
                            print(f"[layout] Outline drawing failed for gid={gid}: {e}")
                            continue

                    if group_label is not None:
                        if merged.geom_type == "Polygon":
                            label_poly = merged
                        else:
                            label_poly = max(merged.geoms, key=lambda g: g.area)

                        bbox = label_poly.bounds
                        text_x, text_y = find_text_position(label_poly, bbox)
                        
                        label = str(group_label)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fs = 0.4
                        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
                        
                        org_x = int(text_x - tw / 2)
                        org_y = int(text_y + th / 2)
                        cv2.putText(canvas_line, label, (org_x, org_y), font, fs, (0, 0, 0), 1, cv2.LINE_AA)
                        
                        vector_elements.append({
                            'type': 'label',
                            'gid': gid,
                            'text': label,
                            'x': text_x,
                            'y': text_y
                        })
                        
                else:
                    print(f"[layout] No valid polygons for gid={gid}")

            except GEOSException as e:
                print(f"[layout] GEOSException for gid={gid}: {e}")
                # Fallback: dung polygon au tien
                if polys:
                    first_poly = make_valid(polys[0])
                    if not first_poly.is_empty:
                        # Ve outline cho first_poly...
                        pass
            except Exception as e:
                print(f"[layout] Unexpected error for gid={gid}: {e}")

    return canvas_fill, canvas_line, vector_elements, bleed_contours_by_gid
