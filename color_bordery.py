import cv2
import numpy as np
from pathlib import Path


def detect_vertical_horizontal_lines(img_bgr,
                                     canny1=80, canny2=200,
                                     min_frac=0.2,
                                     min_len_frac=0.25,
                                     max_gap=5):
    """
    Trả về list line dọc & ngang dạng (x1, y1, x2, y2)
    chỉ giữ line gần như dọc / ngang & đủ dài.
    """
    h, w = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Edge map
    edges = cv2.Canny(gray, canny1, canny2)

    # Hough transform tìm line thẳng
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(min(h, w) * min_len_frac),
        maxLineGap=max_gap
    )

    vertical, horizontal = [], []
    if lines is None:
        return vertical, horizontal

    for l in lines[:, 0]:
        x1, y1, x2, y2 = map(int, l)
        dx = x2 - x1
        dy = y2 - y1

        # Gần như dọc
        if abs(dx) < abs(dy) * min_frac:
            # Chuẩn hoá: từ trên xuống dưới
            if y2 < y1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            vertical.append((x1, y1, x2, y2))

        # Gần như ngang
        elif abs(dy) < abs(dx) * min_frac:
            # Chuẩn hoá: trái sang phải
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            horizontal.append((x1, y1, x2, y2))

    return vertical, horizontal


def draw_lines_on_image(img_bgr, vertical, horizontal,
                        color_v=(0, 0, 255),   # đỏ
                        color_h=(0, 255, 0),   # xanh lá
                        thickness=2):
    """
    Vẽ line dọc/ngang lên ảnh copy, trả về ảnh mới.
    """
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in vertical:
        cv2.line(out, (x1, y1), (x2, y2), color_v, thickness, cv2.LINE_AA)
    for (x1, y1, x2, y2) in horizontal:
        cv2.line(out, (x1, y1), (x2, y2), color_h, thickness, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    in_path = "29.jpg"
    out_path = "29_with_lines.png"

    img = cv2.imread(in_path)
    if img is None:
        raise FileNotFoundError(in_path)

    vertical, horizontal = detect_vertical_horizontal_lines(img)

    print(f"Vertical: {len(vertical)}, Horizontal: {len(horizontal)}")

    img_out = draw_lines_on_image(img, vertical, horizontal)

    cv2.imwrite(out_path, img_out)
    print("Saved:", out_path)
