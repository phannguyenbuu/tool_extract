from ultralytics import SAM
import cv2
import numpy as np

# ===== CONFIG =====
IMG_PATH = "29.jpg"
MODEL_PATH = "sam_b.pt"
TILE_SIZE = 256      # kích thước tile (px)
MIN_AREA = 30        # bỏ polygon quá nhỏ

# ===== LOAD =====
model = SAM(MODEL_PATH)
img = cv2.imread(IMG_PATH)
H, W = img.shape[:2]

all_polys = []

# ===== TILING + SAM =====
for y0 in range(0, H, TILE_SIZE):
    for x0 in range(0, W, TILE_SIZE):
        y1 = min(y0 + TILE_SIZE, H)
        x1 = min(x0 + TILE_SIZE, W)

        tile = img[y0:y1, x0:x1]

        # SAM nhận numpy BGR được luôn [web:65][web:121]
        results = model.predict(
            source=tile,
            device="cpu",
            imgsz=TILE_SIZE,
        )

        r = results[0]
        if r.masks is None:
            continue

        # r.masks.xy: list [(N_i, 2), ...] toạ độ trong tile
        for poly in r.masks.xy:
            pts = poly.astype(np.float64)

            # dịch toạ độ về ảnh gốc
            pts[:, 0] += x0
            pts[:, 1] += y0

            pts_int = pts.astype(np.float64)
            area = cv2.contourArea(pts_int)
            if area < MIN_AREA:
                continue

            all_polys.append(pts_int)

print("Total polygons:", len(all_polys))

# ===== DRAW =====
overlay = img.copy()
for pts in all_polys:
    pts_ = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts_], True, (0, 255, 0), 1)
    cv2.fillPoly(overlay, [pts_], (0, 255, 0))

alpha = 0.25
out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
cv2.imwrite("29_sam_tiles.jpg", out)
print("Saved to 29_sam_tiles.jpg")
