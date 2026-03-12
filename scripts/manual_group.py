# manual_group.py
import json
from typing import Dict, List, Tuple

def load_manual_groups(path: str, num_polygons: int) -> Tuple[Dict[int, int], List[List[int]]]:
    """
    path: uong dan group.json
    num_polygons: tong so polygon
    Tra ve:
      poly_to_group: map poly_idx -> group_id
      all_groups: list[list[poly_idx]]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_groups: List[List[int]] = []
    poly_to_group: Dict[int, int] = {}

    print("DEBUG JSON[0]:", data[0]) 

    # 1) group theo json
    for gid, grp in enumerate(data):
        clean = sorted({int(i) for i in grp if 0 <= int(i) < num_polygons})
        if not clean:
            continue
        all_groups.append(clean)
        for idx in clean:
            poly_to_group[idx] = gid

    # 2) phan con lai: moi polygon la 1 group rieng
    # remaining = [i for i in range(num_polygons) if i not in poly_to_group]
    # base_gid = len(all_groups)
    # for k, idx in enumerate(remaining):
    #     gid = base_gid + k
    #     all_groups.append([idx])
    #     poly_to_group[idx] = gid

    print("So group theo json khong co manh le:", len(all_groups))
    return poly_to_group, all_groups
