from lxml import etree
from svgpathtools import parse_path
import numpy as np

import re
from lxml import etree

def parse_transform(transform_str, default=(0,0)):
    """Parse translate(a b) → (a,b); scale đơn giản ignore vì flip Y không ảnh hưởng pos"""
    if not transform_str:
        return default
    # Extract translate values
    match = re.search(r'translate\(([^,\s]+)[,\s]+([^)\s]+)\)', transform_str)
    if match:
        tx, ty = float(match.group(1)), float(match.group(2))
        return (tx, ty)
    return default

def svg_text_center(element):
    """Center từ transform của <text> + tspan x,y"""
    transform = element.get('transform', '')
    base_pos = parse_transform(transform)
    
    # Lấy first tspan x,y (relative to text)
    tspans = element.xpath('.//svg:tspan', namespaces={'svg': 'http://www.w3.org/2000/svg'})
    if tspans:
        dx = float(tspans[0].get('x', 0))
        dy = float(tspans[0].get('y', 0))
    else:
        dx = dy = 0
    
    x = base_pos[0] + dx
    y = base_pos[1] + dy
    return np.array([x, y])


def sample_path_points(path_str, num_samples=200):
    if not path_str:
        return np.empty((0, 2))
    path = parse_path(path_str)
    if path.length() == 0:
        return np.empty((0, 2))
    ts = np.linspace(0, 1, num_samples)
    points_complex = np.array([path.point(t) for t in ts])
    # Convert complex -> (N,2) real/imag
    points = np.column_stack((points_complex.real, points_complex.imag))
    return points

def get_svg_text(element):
    """Lấy full text từ <text> bao gồm tất cả <tspan> con, recursive"""
    texts = []
    if element.text:
        texts.append(element.text.strip())
    for child in element.iterchildren():
        if child.tag.endswith('tspan'):
            texts.append(get_svg_text(child))
        elif child.text:
            texts.append(child.text.strip())
    if element.tail:
        texts.append(element.tail.strip())
    return ' '.join(t for t in texts if t)

def point_in_path(center, path_points):
    """Kiểm tra point có nằm trong closed path points (ray casting)"""
    x, y = center
    n = len(path_points)
    inside = False
    p1x, p1y = path_points[0]
    for i in range(n + 1):
        p2x, p2y = path_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def extract_texts_by_path_polygon(svg_file_path, samples_per_path=50):
    tree = etree.parse(svg_file_path)
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    
    path_elements = tree.xpath('//svg:path[@d]', namespaces=ns)
    path_data = {}
    for i, path_el in enumerate(path_elements):
        path_str = path_el.get('d', '').strip()
        path_id = path_el.get('id', f'path-{i}')
        points = sample_path_points(path_str, samples_per_path)
        if len(points) > 5:  # Min points cho polygon
            path_data[path_id] = {'element': path_el, 'points': points}
    
    texts_elements = tree.xpath('//svg:text', namespaces=ns)
    text_centers = []
    for t in texts_elements:
        full_text = get_svg_text(t)
        if full_text:
            center = svg_text_center(t)
            text_centers.append((full_text, center))
    
    # Group: text center INSIDE path polygon
    results = []
    total = len(path_data.items())


    txt = '['

    for idx, (path_id, data) in enumerate(path_data.items()):
        
        group_texts = []
        for text_val, center in text_centers:
            if point_in_path(center, data['points']):
                group_texts.append(text_val)
        if group_texts:
            txt += f'   [{",".join(group_texts)}],\n'
            print(f'{idx}/{total}:{path_id}:{",".join(group_texts)}')
            results.append({'path_id': path_id, 'texts': group_texts})

    txt = txt.rstrip(',\n') + '\n]'  # Bỏ dấu , cuối
    
    # Save
    with open('tool_extract/group.json', 'w', encoding='utf-8') as f:
        f.write(txt)
    
    return path_data, text_centers, results



import matplotlib.pyplot as plt

def plot_svg_debug(svg_file_path, path_data, text_centers, groups=None, tolerance=2.0):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(path_data)))
    color_idx = 0
    
    # Vẽ path points (mỗi path 1 màu)
    for path_id, data in path_data.items():
        points = data['points']
        if len(points) > 0:
            ax.scatter(points[:,0], points[:,1], s=2, c='red', alpha=1, label=f'Path {path_id} points')
            ax.plot(points[:,0], points[:,1], c='red', alpha=1, linewidth=0.1)
            # color_idx += 1
    
    # Vẽ text centers (X đỏ)
    text_xs = [center[0] for _, center in text_centers]
    text_ys = [center[1] for _, center in text_centers]
    ax.scatter(text_xs, text_ys, c='blue', s=100, marker='X', linewidth=1, label='Text centers', zorder=5)
    
    # Annotate text values
    # for i, (text_val, center) in enumerate(text_centers):
    #     ax.annotate(text_val, (center[0], center[1]), xytext=(3, 3), 
    #                textcoords='offset points', fontsize=9, fontweight='bold',
    #                bbox=dict(boxstyle='round,pad=0.1', facecolor='yellow', alpha=0.7))
    
    # Highlight grouped texts nếu có groups
    # if groups:
    #     for group in groups:
    #         path_id = group['path_id']
    #         if path_id in path_data:
    #             color = colors[list(path_data.keys()).index(path_id) % len(colors)]
    #             for text_val in group['texts']:
    #                 # Tìm center của text này
    #                 for tval, center in text_centers:
    #                     if tval == text_val:
    #                         circle = plt.Circle((center[0], center[1]), tolerance, 
    #                                           color=color, fill=False, linewidth=3, linestyle='--')
    #                         ax.add_patch(circle)
    #                         break
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    ax.invert_yaxis()  # SVG y-axis xuống dưới
    ax.set_title('SVG Debug: Path points (colored lines/dots) vs Text centers (red X + labels)\nYellow boxes: all texts | Dashed circles: grouped by tolerance')
    plt.tight_layout()
    plt.show()

# Sử dụng ngay sau extract (trong main)
# groups = extract_texts_by_path_points(svg_file)
# plot_svg_debug(svg_file, path_data, text_centers, groups)  # path_data, text_centers từ function

import json

# Sử dụng
svg_file = r'tool_extract\raw_01.svg'  # File SVG của bạn
path_data, text_centers, groups = extract_texts_by_path_polygon(svg_file)






# plot_svg_debug(svg_file, path_data, text_centers, groups)

grouped_indices = set()
for group in groups:
    text_to_idx = {text: i for i, (text, _) in enumerate(text_centers)}
    for text in group['texts']:
        grouped_indices.add(text_to_idx[text])

total_texts = len(text_centers)
missed_indices = [i for i in range(total_texts) if i not in grouped_indices]
missed_texts = [text_centers[i][0] for i in missed_indices]

print(f"Total texts: {total_texts}")
# print(f"Grouped: {len(grouped_indices)} ({len(grouped_indices)/total_texts*100:.1f}%)")
# print(f"Missed: {len(missed_indices)} ({len(missed_indices)/total_texts*100:.1f}%)")
# print("Missed indices:", missed_indices)
print("Missed texts sample:", missed_texts[:10])  # 10 đầu