import json
import re
from xml.etree import ElementTree as ET

# 1. Đọc mapping
with open('random_mapping_0_108.json', 'r') as f:
    mapping = json.load(f)

# 2. Parse SVG với ElementTree (GIỮ ATTRIBUTES)
tree = ET.parse('cho_ben_thanh_v3.svg')
root = tree.getroot()

# Namespace
namespaces = {'svg': 'http://www.w3.org/2000/svg'}

changed = 0
for text_elem in root.findall('.//svg:text', namespaces):
    # 🔍 Duyệt tất cả TEXT NODES và TSPANS
    for child in list(text_elem):  # list() để tránh modification during iteration
        if child.text and child.text.strip().isdigit():
            old_num = int(child.text.strip())
            if 0 <= old_num <= 108:
                new_num = mapping[str(old_num)]
                print(f"🔄 [{old_num} → {new_num}] tại vị trí: x={child.get('x','?')} y={child.get('y','?')}")
                
                # ✅ CHỈ THAY TEXT CONTENT, GIỮ 100% ATTRIBUTES
                child.text = str(new_num)
                changed += 1
    
    # Nếu text trực tiếp trong <text> (không có tspan)
    if text_elem.text and text_elem.text.strip().isdigit():
        old_num = int(text_elem.text.strip())
        if 0 <= old_num <= 108:
            new_num = mapping[str(old_num)]
            print(f"🔄 [{old_num} → {new_num}] text trực tiếp")
            text_elem.text = str(new_num)
            changed += 1

print(f"✅ Thay đổi {changed} text elements - GIỮ NGUYÊN VỊ TRÍ!")

# 3. Lưu với PRETTY PRINT + GIỮ ATTRIBUTES
ET.indent(tree, space="  ", level=0)
tree.write('cho_ben_thanh_v3_remapped_PERFECT.svg', 
           encoding='utf-8', 
           xml_declaration=True)

print("🎉 Đã lưu 'cho_ben_thanh_v3_remapped_PERFECT.svg'")
