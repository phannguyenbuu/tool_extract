import cv2
import numpy as np
import sys
import os
from sklearn.linear_model import RANSACRegressor

def fit_straight_lines(contour, min_inliers=10, max_lines=8):
    """RANSAC fit lines từ contour points"""
    points = contour.squeeze()
    if len(points) < min_inliers * 2: return []
    
    lines = []
    used = np.zeros(len(points), bool)
    
    for _ in range(max_lines):
        if np.sum(~used) < min_inliers: break
        
        X = points[~used, 0]
        Y = points[~used, 1]
        
        # RANSAC fit y = mx + c
        model = RANSACRegressor(residual_threshold=3.0, min_samples=5)
        model.fit(X.reshape(-1,1), Y)
        inliers = model.inlier_mask_
        
        if np.sum(inliers) >= min_inliers:
            m = model.estimator_.coef_[0]
            c = model.estimator_.intercept_
            x1, y1 = X[inliers].min(), int(m * X[inliers].min() + c)
            x2, y2 = X[inliers].max(), int(m * X[inliers].max() + c)
            lines.append([(int(x1), int(y1)), (int(x2), int(y2))])
            
            # Mark used
            idx = np.where(~used)[0][inliers]
            used[np.where(~used)[0][inliers]] = True
    
    return lines

def intersect_lines(lines):
    """Tìm giao điểm lines → nodes"""
    nodes = []
    for i, l1 in enumerate(lines):
        for j, l2 in enumerate(lines[i+1:], i+1):
            p1, p2 = l1
            p3, p4 = l2
            
            denom = (p1[0]-p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]-p4[0])
            if abs(denom) < 1e-6: continue
            
            px = ((p1[0]*p2[1]-p1[1]*p2[0])*(p3[0]-p4[0]) - 
                  (p1[0]-p2[0])*(p3[0]*p4[1]-p3[1]*p4[0])) / denom
            py = ((p1[0]*p2[1]-p1[1]*p2[0])*(p3[1]-p4[1]) - 
                  (p1[1]-p2[1])*(p3[0]*p4[1]-p3[1]*p4[0])) / denom
            
            nodes.append((int(px), int(py)))
    
    # Cluster nodes gần nhau
    nodes = np.array(nodes)
    clustered = []
    for pt in nodes:
        if not clustered or not any(np.linalg.norm(pt - c) < 8 for c in clustered):
            clustered.append(pt)
    
    return clustered


def save_svg_contours(all_contours_data, filename="nodes_map.svg", width=1000, height=800):
    """Xuất SVG từ tất cả contours + nodes"""
    if not all_contours_data:
        print("Không có contours để xuất SVG")
        return
    
    # Tìm bounds tổng
    all_points = []
    for _, approx, bbox in all_contours_data:
        all_points.extend(approx)
        x,y,w,h = bbox
        all_points.extend([[x,y], [x+w,y], [x,y+h], [x+w,y+h]])
    
    all_points = np.array(all_points)
    margin = 40
    min_x, min_y = all_points.min(axis=0) - margin
    max_x, max_y = all_points.max(axis=0) + margin
    svg_w = max_x - min_x
    svg_h = max_y - min_y
    
    scale_x = width / svg_w
    scale_y = height / svg_h
    scale = min(scale_x, scale_y)
    
    svg = f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" 
              xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#f8f9fa"/>
    '''
    
    for idx, (status, approx, bbox) in enumerate(all_contours_data):
        color = "#28a745" if status == "Real" else "#dc3545"
        stroke_width = 2.5 if status == "Real" else 1.5
        
        # Transform points
        approx_svg = []
        for point in approx:
            px = (point[0] - min_x) * scale
            py = (point[1] - min_y) * scale
            approx_svg.append((px, py))
        
        # Vẽ contour path (nối các nodes)
        path_data = f"M {approx_svg[0][0]:.1f},{approx_svg[0][1]:.1f} "
        for px, py in approx_svg[1:]:
            path_data += f"L {px:.1f},{py:.1f} "
        path_data += "Z"
        
        svg += f'<path d="{path_data}" fill="none" stroke="{color}" '
        svg += f'stroke-width="{stroke_width}" stroke-linejoin="round" opacity="0.8"/>\n'
        
        # Vẽ nodes
        for j, (px, py) in enumerate(approx_svg):
            svg += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{color}" '
            svg += 'stroke="#ffffff" stroke-width="2.5"/>\n'
            svg += f'<text x="{px+8:.1f}" y="{py+3:.1f}" font-size="11" '
            svg += f'font-weight="bold" fill="#333">{j}</text>\n'
        
        # Label status
        x,y,w,h = bbox
        lx = (x - min_x) * scale + 5
        ly = (y - min_y) * scale - 10
        svg += f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="14" font-weight="bold" '
        svg += f'fill="{color}">{status}</text>\n'
    
    svg += '</svg>'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(svg)
    print(f"✅ Xuất SVG: {filename} ({len(all_contours_data)} contours)")



def preprocess_image(src):
    """Simply màu + gộp đỉnh TRƯỚC gray"""
    
    # 1. QUANTIZE MÀU (giảm 256→32 màu)
    data = np.float32(src).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(data, 32, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Áp centers về uint8
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    src_quant = quantized.reshape(src.shape)
    
    # 2. GỘP MÀU GẦN (median 3x3)
    src_simple = cv2.medianBlur(src_quant, 3)
    
    # 3. MORPH CLOSE gộp vùng nhỏ
    kernel = np.ones((3,3), np.uint8)
    src_closed = cv2.morphologyEx(src_simple, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 4. Bilateral giữ cạnh
    bilateral = cv2.bilateralFilter(src_closed, 5, 30, 30)
    
    print("✅ Preprocess: quantize→median→close→bilateral")
    return bilateral


def detect_objects(image_path):
    # Đọc ảnh
    src = cv2.imread(image_path)
    
    

    bilateral = preprocess_image(src)
    gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)  # GIỮ GRAY ỔN
    
    canny = cv2.Canny(gray, 30, 80, apertureSize=3)
    
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = [c for c in contours if cv2.contourArea(c) > 20]
    
    print(f"Tìm {len(contours)} segments từ Canny")
    
    all_lines = []
    for i, contour in enumerate(contours[:20]):  # Top 20 segments
        lines = fit_straight_lines(contour)
        all_lines.extend(lines)
        print(f"Segment {i}: {len(lines)} lines fitted")
    
    # Vẽ lines sắc nét
    src_result = src.copy()
    for line in all_lines:
        cv2.line(src_result, line[0], line[1], (0, 255, 255), 3)  # Cyan thick
    
    # Nội suy nodes từ intersections
    nodes = intersect_lines(all_lines)

    all_contours_data = []

    for i, c in enumerate(contours):
        epsilon = 0.015 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        corners = len(approx)
        
        status = "Real" if corners >= 5 else "Fake"
        color = (0, 255, 0) if status == "Real" else (0, 0, 255)
        
        x,y,w,h = cv2.boundingRect(c)
        all_contours_data.append((status, approx, (x,y,w,h)))
        
        # ✅ KIỂM TRA GÓC LÕM <10°
        sharp_angles = 0
        sharp_points = []
        
        if len(approx) >= 3:
            pts = approx.reshape(-1, 2).astype(np.float32)
            for j in range(len(pts)):
                p1 = pts[(j-1) % len(pts)]
                p2 = pts[j]
                p3 = pts[(j+1) % len(pts)]
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Góc nội (0-180°)
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                
                if angle < 10:  # ⚠️ GÓC LÕM NHỌN
                    sharp_angles += 1
                    sharp_points.append((j, angle))
        
        area = cv2.contourArea(c)
        
        # CẢNH BÁO nếu có góc <10°
        warning = f"⚠️{sharp_angles}g<10°" if sharp_angles > 0 else f"OK A:{area:.0f}"
       
        # Highlight sharp corners
        for j, angle in sharp_points:
            pt = tuple(approx[j][0])
            cv2.circle(src, pt, 8, (0, 0, 255), 2)  # Đỏ viền nổi
            # cv2.putText(src, f"{angle:.0f}°", (pt[0]+10, pt[1]+25),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        cv2.drawContours(src, [c], 0, (80, 80, 80), 5)
        
        # Print cảnh báo
        if sharp_angles > 0:
            print(f"⚠️  {i}: {corners}g | {sharp_angles} GÓC LÕM <10° → {sharp_points}")
        else:
            print(f"   {i}: {corners}g | OK")


    # save_svg_contours(all_contours_data)

    # Hiển thị
    cv2.imshow("Original + Contours", src)
    # cv2.imshow("Canny Edges", thr)
    cv2.imshow("Gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("result.jpg", src)
    print("Lưu result.jpg")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        detect_objects(sys.argv[1])
    else:
        detect_objects("src.jpg")
