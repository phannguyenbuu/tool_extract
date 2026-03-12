# better_polygonize_offset.py
import os
from collections import defaultdict

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from polygonize_utils import load_polylines, build_segments, auto_join_gaps, polygonize_segments
from manual_group import load_manual_groups
from regions_build import build_regions, cad_to_px_builder
from layout_pack import build_layout_canvas
from export_svg import export_hybrid_svg, export_line_svg, export_orig_svg
from group_patch import get_group_to_regions

def draw_original_layout_groups(regions, base_image_shape,
                                bg_color=(255, 255, 255),
                                buf_px=2.0):
    """
    buf_px: offset (px) dung cho buffer truoc/ sau khi union.
    """
    h, w = base_image_shape[:2]
    canvas = np.full((h, w, 3), bg_color, np.uint8)

    group_to_polys = defaultdict(list)
    for r in regions:
        gid = r.get("group", -1)
        if gid < 0:
            continue
        pts = np.array(r["points"], dtype=np.float64)
        if len(pts) < 3:
            continue
        poly = Polygon(pts)
        if poly.is_empty:
            continue
        # no ra buf_px e dinh cac khe ho/inh
        poly = poly.buffer(buf_px)
        if poly.is_empty:
            continue
        group_to_polys[gid].append(poly)


    group_to_label = {}
    for r in regions:
        gid = r.get("group", -1)
        lab = r.get("label", 0)
        if gid >= 0 and lab > 0:
            group_to_label[gid] = lab

    for gid, polys in group_to_polys.items():
        if not polys:
            continue

        merged = unary_union(polys)
        if merged.is_empty:
            continue

        # co lai cung luong e tra ve gan bien that
        merged = merged.buffer(-buf_px)

        if merged.geom_type == "Polygon":
            geoms = [merged]
        else:
            geoms = list(merged.geoms)

        # ve outline group
        for g in geoms:
            coords = np.array(g.exterior.coords, dtype=np.int32)
            cv2.polylines(canvas, [coords], True, (0, 0, 0), 2, cv2.LINE_AA)

        # centroid union cho so group
        c = merged.centroid

        if c.is_empty:
            continue

        cx, cy = float(c.x), float(c.y)
        # label = str(gid)      # hoac gid+1
        
        label = str(group_to_label.get(gid, gid + 1))


        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.6
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        org = (int(cx - tw / 2), int(cy + th / 2))
        cv2.putText(canvas, label, org, font, fs, (0, 0, 255), 1, cv2.LINE_AA)

    return canvas


def polygonize_offset(JSON_FILE, IMG_SRC, output_path_base):
    lines = load_polylines(JSON_FILE)
    segments, bbox = build_segments(lines, snap=0.2)
    segments, n_added = auto_join_gaps(segments, join_dist=2.0)
    polygons = polygonize_segments(segments)

    num_polygons = len(polygons)
    group_json = "group.json"  # hoac nhan tu argv

    # if os.path.exists(group_json):
    #     print('Start read json group...')
    poly_to_group, all_groups = load_manual_groups(group_json, num_polygons)
    # else:
    # poly_to_group, all_groups = group_component_polygons(polygons)

    # poly_to_group, all_groups = group_component_polygons(polygons)
    print("Tong group sau cung:", len(all_groups))

    # print("poly_to_group (first 20):", {i: poly_to_group.get(i) for i in range(20)})
    # print("group 0 (from JSON):", sorted([i for i,g in poly_to_group.items() if g == 0]))


    print_debug(polygons,poly_to_group, bbox)



    regions, img_out, _ = build_regions(polygons, poly_to_group, bbox, IMG_SRC)
    # regions = unify_close_vertices(regions, thr_x=20, thr_y=20)

    group_to_regions = get_group_to_regions(regions)

    # from export_svg import normalize_group_polys  # Reuse
    # group_to_clean_polys = {}
    # for gid, regs in group_to_regions.items():
    #     polys = [Polygon(r["pts"]) for r in regs if len(r["pts"]) >= 3]
    #     if polys:
    #         clean_geoms = normalize_group_polys(polys)
    #         group_to_clean_polys[gid] = clean_geoms

    # print(f'Union early: {len(group_to_regions)}  {len(group_to_clean_polys)} groups')

    # # Convert cho rotate/pack
    # group_to_regions_clean = {}
    # for gid, clean_polys in group_to_clean_polys.items():
    #     regs_clean = [{"pts": np.array(p.exterior.coords),
    #                 "centroid": (p.centroid.x, p.centroid.y),
    #                 "label": gid, "group": gid, "color": (200,200,200)}
    #                 for p in clean_polys]
    #     group_to_regions_clean[gid] = regs_clean


    canvas_fill, canvas_line, vector_elements, bleed_contours = build_layout_canvas(group_to_regions, max_width=2954)
    h, w = canvas_fill.shape[:2]

    out_fill_png  = output_path_base + ".png"
    out_line_png  = output_path_base + "_line.png"
    out_orig_png  = output_path_base + "_orig.png"

    out_hybrid_svg = output_path_base + "_hybrid.svg"
    out_line_svg   = output_path_base + "_line.svg"
    out_orig_svg   = output_path_base + "_orig.svg"

    # PNG (neu van muon)
    cv2.imwrite(out_fill_png, canvas_fill)
    cv2.imwrite(out_line_png, canvas_line)
    canvas_orig = draw_original_layout_groups(regions, img_out.shape)
    cv2.imwrite(out_orig_png, canvas_orig)

    # SVG
    export_hybrid_svg(canvas_fill, vector_elements, bleed_contours, w, h, out_hybrid_svg)
    export_line_svg(vector_elements, w, h, out_line_svg)
    export_orig_svg(regions, img_out.shape[1], img_out.shape[0], out_orig_svg)

    return out_fill_png, out_line_png, out_orig_png, out_hybrid_svg, out_line_svg, out_orig_svg




def unify_close_vertices(regions,
                         thr_x: float = 0.5,
                         thr_y: float = 0.5):
    """
    Gom cac inh gan nhau ve cung 1 toa o.
    - thr_x, thr_y: nguong lech toi a theo truc (on vi pixel sau cad_to_px).
    """
    # thu tat ca iem
    all_pts = []
    for r in regions:
        pts = r.get("points")
        if not pts:
            continue
        for x, y in pts:
            all_pts.append((float(x), float(y)))

    if not all_pts:
        return regions

    all_pts = np.asarray(all_pts, dtype=np.float64)

    # gom cluster bang thuat toan on gian O(n^2) (so inh khong qua lon)
    n = len(all_pts)
    visited = np.zeros(n, dtype=bool)
    clusters = []  # list[list[index]]

    for i in range(n):
        if visited[i]:
            continue
        cx, cy = all_pts[i]
        cluster = [i]
        visited[i] = True
        # gom cac iem nam trong cua so thr_x, thr_y
        for j in range(i + 1, n):
            if visited[j]:
                continue
            x, y = all_pts[j]
            if abs(x - cx) <= thr_x and abs(y - cy) <= thr_y:
                cluster.append(j)
                visited[j] = True
        clusters.append(cluster)

    # tinh toa o hop nhat (o ay dung trung binh)
    unified = np.zeros_like(all_pts)
    for cl in clusters:
        pts_cl = all_pts[cl]
        ux, uy = pts_cl.mean(axis=0)
        unified[cl] = (ux, uy)

    # ghi nguoc vao regions
    k = 0
    for r in regions:
        pts = r.get("points")
        if not pts:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        m = len(arr)
        arr[:, 0] = unified[k:k + m, 0]
        arr[:, 1] = unified[k:k + m, 1]
        r["points"] = arr.tolist()
        k += m

    return regions



def print_debug(polygons,poly_to_group, bbox):
        # --- debug: ve UNION cua TUNG GROUP tu poly_to_group ---
    

        # --- debug 1: CAC POLYGON GOC (truoc group) ---
    xmin, ymin, xmax, ymax = bbox
    margin = 20
    scale = 1.0
    width  = int((xmax - xmin) * scale) + 2 * margin
    height = int((ymax - ymin) * scale) + 2 * margin
    width  = max(width, 1000)
    height = max(height, 1000)

    cad_to_px = cad_to_px_builder(xmin, xmax, ymin, ymax, scale, margin)

    # POLYGON GOC - mau o nhat, khong label
    # --- debug 1: CAC POLYGON GOC (truoc group) - in INDEX POLYGON ---
    polygons_canvas = np.full((height, width, 3), 255, np.uint8)
    for idx, poly in enumerate(polygons):
        # if poly.area < 100:
            # continue

        coords = []
        for x, y in poly.exterior.coords:
            px, py = cad_to_px(x, y)
            coords.append((px, py))
        pts = np.array(coords, dtype=np.int32)
        

        # VIEN mong (optional)
        cv2.polylines(polygons_canvas, [pts], True, (50, 50, 50), 1, cv2.LINE_AA)



        # if idx == 80 or idx == 309:
        #     print(f"Polygon {idx} area: {poly.area:.2f}")
        
        # LABEL: index cua polygon (khong phai group)
        cx, cy = poly.centroid.x, poly.centroid.y
        
        cx_px, cy_px = cad_to_px(cx, cy)
        label = str(idx)  #  INDEX POLYGON thay vi group_id
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.4
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        org = (int(cx_px - tw / 2), int(cy_px + th / 2))
        cv2.putText(polygons_canvas, label, org, font, fs, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite("debug_polygons_raw.png", polygons_canvas)


    # --- debug 2: UNION CUA TUNG GROUP (sau group) ---
    debug_canvas = np.full((height, width, 3), 255, np.uint8)

    # Gom polygon theo group
    group_to_polys = defaultdict(list)
    for idx, poly in enumerate(polygons):
        gid = poly_to_group.get(idx, -1)
        if gid >= 0:
            group_to_polys[gid].append(poly)

    # Ve tung group union (xanh am + label o)
    for gid, polys in group_to_polys.items():
        if not polys:
            continue
        
        polys_clean = []
        for p in polys:
            if p.is_empty:
                continue
            p_clean = make_valid(p)  # Fix invalid
            if p_clean.is_empty:
                continue
            polys_clean.append(p_clean)

        if not polys_clean:
            continue  # Skip group rong

        try:
            merged = unary_union(polys_clean)
        except Exception as e:
            print(f"[WARN] unary_union fail gid={gid}: {e}")
            # Fallback: dung polygon au tien
            merged = polys_clean[0]
            if merged.geom_type != 'Polygon':
                continue

        if merged.is_empty:
            continue

        merged = merged.buffer(0.5)
        merged = merged.buffer(-0.5)

        if merged.geom_type == "Polygon":
            geoms = [merged]
        else:
            geoms = list(merged.geoms)

        for geom in geoms:
            coords = []
            for x, y in geom.exterior.coords:
                px, py = cad_to_px(x, y)
                coords.append((px, py))
            pts = np.array(coords, dtype=np.int32)
            cv2.polylines(debug_canvas, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)

        c = merged.centroid
        cx_px, cy_px = cad_to_px(float(c.x), float(c.y))
        label = str(gid)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.6
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        org = (int(cx_px - tw / 2), int(cy_px + th / 2))
        cv2.putText(debug_canvas, label, org, font, fs, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite("debug_group_unions.png", debug_canvas)



if __name__ == "__main__":
    polygonize_offset("polylines.json", "input.png", "groups_layout_binpack_bbox.png")
