
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from shapely.geometry import Polygon, Point

def find_text_position(poly: Polygon, bbox: tuple) -> Tuple[float, float]:
    """
    Quet grid 10px x 10px trong bbox, tim iem au tien KHONG nam trong polygon 
    (outside/edge cho label ro). Neu toan bo trong  middle_x, top-5px.
    """
    xmin, ymin, xmax, ymax = bbox
    step = 10.0  # Grid 10px
    
    # 1) Grid scan: tu leftright, topbottom
    for grid_y in range(int(ymin), int(ymax+1), int(step)):
        gy = float(grid_y)
        for grid_x in range(int(xmin), int(xmax+1), int(step)):
            gx = float(grid_x)
            pt = Point(gx, gy)
            # Outside hoac boundary  dung ngay (label ro)
            if not poly.contains(pt) and not poly.interiors:  # Skip hole interiors
                return gx, gy
    
    # 2) Fallback: middle_x, top-5px (vua tren bbox)
    mx = 0.5 * (xmin + xmax)
    ty = ymax - 5.0
    return mx, ty



def extract_group_patch(regs):
    # gom tat ca iem cua group e lay bbox
    all_pts = np.concatenate(
        [np.array(r["points"], dtype=np.float64) for r in regs],
        axis=0
    )
    x_min = all_pts[:, 0].min()
    x_max = all_pts[:, 0].max()
    y_min = all_pts[:, 1].min()
    y_max = all_pts[:, 1].max()

    w = x_max - x_min + 1
    h = y_max - y_min + 1

    # patch nen trang
    patch = np.full((h, w, 3), 255, np.uint8)

    # copy polygon + label vao patch, ong thoi ua toa o ve local
    local_regions = []
    for r in regs:
        pts = np.array(r["points"], dtype=np.float64)
        pts_local = pts.copy()
        pts_local[:, 0] -= x_min
        pts_local[:, 1] -= y_min

        color = tuple(map(int, r["color"]))  # BGR
        cv2.fillPoly(patch, [pts_local], color)

        cx = np.mean(pts_local[:, 0])
        cy = np.mean(pts_local[:, 1])

        lxmin,lxmax = pts_local[:,0].min(), pts_local[:,0].max()
        lymin,lymax = pts_local[:,1].min(), pts_local[:,1].max()
        local_bbox = (lxmin, lymin, lxmax, lymax)

        tx, ty = find_text_position(Polygon(pts_local), local_bbox)

        local_regions.append({
            "pts": pts_local,
            "label": r["label"],
            "centroid": (tx, ty),
            "color": color
        })


    return patch, local_regions, (w, h)


def get_group_to_regions(regions: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    group_to_regions: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in regions:
        gid = r.get("group", -1)
        if gid < 0:
            continue
        # am bao co "pts" va "centroid"
        if "pts" not in r:
            pts = np.array(r["points"], dtype=np.float64)
            r["pts"] = pts
        if "centroid" not in r:
            pts = r["pts"]
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            r["centroid"] = (cx, cy)
        group_to_regions[gid].append(r)
    return group_to_regions



def rotate_regions_once(local_regions, angle_deg):
    # TRUE CENTROID thay bbox center
    all_pts = np.concatenate([r["pts"] for r in local_regions], axis=0)
    true_center = np.mean(all_pts, axis=0)  # Mean tat ca pts  KHONG +1 bias
    
    M = cv2.getRotationMatrix2D(true_center, angle_deg, 1.0)
    
    rot_regions = []
    all_rot_pts = []
    
    for r in local_regions:
        pts = r["pts"].astype(np.float32)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_h = np.hstack([pts, ones])
        pts_rot = (M @ pts_h.T).T
        
        # Rotate centroid TUC
        cx, cy = r["centroid"]
        c_h = np.array([[cx, cy, 1.0]], dtype=np.float32).T
        c_rot = (M @ c_h).T[0]
        
        rot_regions.append({
            "pts": pts_rot,  # GIU float32
            "label": r["label"],
            "centroid": (float(c_rot[0]), float(c_rot[1])),
            "color": r["color"]
        })
        all_rot_pts.append(pts_rot)
    
    all_rot_pts = np.concatenate(all_rot_pts, axis=0)
    rx_min, ry_min = all_rot_pts.min(axis=0)
    rx_max, ry_max = all_rot_pts.max(axis=0)
    rw = rx_max - rx_min  # NO +1  tight bbox
    rh = ry_max - ry_min
    
    return rot_regions, (rw, rh)
