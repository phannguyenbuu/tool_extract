import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Đọc file JSON
with open('polylines.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

polylines = data['polylines']

# Tạo figure với kích thước phù hợp bounds (khoảng 3000x9000)
fig, ax = plt.subplots(figsize=(10, 30))

# Vẽ từng polyline
for i, poly in enumerate(polylines):
    if len(poly) > 1:  # Chỉ vẽ nếu có ít nhất 2 điểm
        xs = [point['x'] for point in poly]
        ys = [point['y'] for point in poly]
        ax.plot(xs, ys, 'k-', linewidth=0.5, alpha=0.8)  # Đường đen mỏng

# Thiết lập bounds tự động, loại bỏ khoảng trắng thừa
ax.autoscale()
ax.set_aspect('equal')  # Giữ tỷ lệ x/y bằng nhau
ax.set_title('Hình vẽ từ polylines.json (271 đường)')
ax.axis('off')  # Ẩn trục tọa độ

plt.tight_layout()
plt.show()

# Lưu file nếu cần
# plt.savefig('polylines_drawn.png', dpi=300, bbox_inches='tight')
