import networkx as nx
import math
from shapely.geometry import Polygon

MAX_GROUP_RADIUS = 180.0  # chinh theo scale ban ve

# ----- 1. GRAPH THEO CANH CHUNG -----

def segments_share_edge(e1, e2, eps=1e-4):
    (x1, y1), (x2, y2) = e1
    (x3, y3), (x4, y4) = e2

    def norm(a, b):
        return (a, b) if a <= b else (b, a)

    a1, a2 = norm((x1, y1), (x2, y2))
    b1, b2 = norm((x3, y3), (x4, y4))

    d11 = math.hypot(a1[0] - b1[0], a1[1] - b1[1])
    d22 = math.hypot(a2[0] - b2[0], a2[1] - b2[1])

    return d11 < eps and d22 < eps   # phai trung ca 2 inh

from collections import defaultdict
from typing import Dict, List, Iterable, Tuple

def split_cluster(indices: Iterable[int], min_size: int, max_size: int) -> List[List[int]]:
    indices = list(indices)
    groups: List[List[int]] = []
    i = 0
    while i < len(indices):
        remain = len(indices) - i
        if remain < min_size and groups:
            groups[-1].extend(indices[i:])
            break
        take = min(max_size, remain)
        groups.append(indices[i:i + take])
        i += take
    return groups


def split_group_by_radius(polygons, group_indices, max_radius):
    """
    group_indices: list[int] cac poly_idx trong 1 group
    max_radius: khoang cach toi a tu centroid cua sub-group toi tung polygon
    """
    group_indices = list(group_indices)
    sub_groups = []

    for idx in group_indices:
        c = polygons[idx].centroid
        cx, cy = c.x, c.y

        placed = False
        for sub in sub_groups:
            # tam hien tai cua sub-group (trung binh centroid)
            sx = sum(polygons[j].centroid.x for j in sub) / len(sub)
            sy = sum(polygons[j].centroid.y for j in sub) / len(sub)
            dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
            if dist <= max_radius:
                sub.append(idx)
                placed = True
                break

        if not placed:
            sub_groups.append([idx])

    return [sorted(sub) for sub in sub_groups]

def cluster_by_vertex(
    polygons,
    refined_components: Iterable[Iterable[int]],
    components: Iterable[Iterable[int]],
    min_size: int = 2,
    max_size: int = 4,
) -> Tuple[Dict[int, int], List[List[int]]]:

    # ----- 3. CHIA NHO MOI CLUSTER THANH GROUP 24 -----
    all_groups: List[List[int]] = []
    for cl in refined_components:
        all_groups.extend(split_cluster(sorted(cl), min_size, max_size))

    # map polygon_index -> group_id (tam)
    poly_to_group: Dict[int, int] = {}
    for gid, grp in enumerate(all_groups):
        for idx in grp:
            poly_to_group[idx] = gid

    print("So group ban au:", len(all_groups))

    # map: polygon_idx -> component_id topo
    poly_to_comp: Dict[int, int] = {}
    for cid, comp in enumerate(components):
        for idx in comp:
            poly_to_comp[idx] = cid

    # group -> tap cac component topo chua trong group o
    group_to_comp: Dict[int, set] = defaultdict(set)
    for idx, gid in poly_to_group.items():
        cid = poly_to_comp.get(idx, -1)
        group_to_comp[gid].add(cid)

    # tao lai all_groups, tach group lai theo component topo
    new_all_groups: List[List[int]] = []
    for gid, grp in enumerate(all_groups):
        comps_in_group = group_to_comp.get(gid, set())
        if len(comps_in_group) <= 1:
            new_all_groups.append(sorted(grp))
        else:
            comp_to_indices: Dict[int, List[int]] = defaultdict(list)
            for idx in grp:
                cid = poly_to_comp.get(idx, -1)
                comp_to_indices[cid].append(idx)
            for sub in comp_to_indices.values():
                if sub:
                    new_all_groups.append(sorted(sub))

    all_groups = new_all_groups

    # ----- 4. TACH THEO BAN KINH CENTROID -----
    

    radius_groups: List[List[int]] = []
    for grp in all_groups:
        radius_groups.extend(split_group_by_radius(polygons, grp, MAX_GROUP_RADIUS))

    all_groups = radius_groups

    # build lai poly_to_group tu all_groups moi nhat
    poly_to_group = {}
    for gid, grp in enumerate(all_groups):
        for idx in grp:
            poly_to_group[idx] = gid

    print("So group sau khi tach group lai + radius:", len(all_groups))

    return poly_to_group, all_groups


def split_component_by_distance(polygons, comp_indices, max_dist):
    comp_indices = list(comp_indices)
    clusters = []
    for idx in comp_indices:
        placed = False
        for cl in clusters:
            # neu centroid gan it nhat 1 polygon trong cluster thi cho vao
            if any(polygons[j].centroid.distance(polygons[idx].centroid) <= max_dist
                for j in cl):
                cl.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
    return [sorted(cl) for cl in clusters]

def group_component_polygons(polygons):
    G = nx.Graph()
    for idx in range(len(polygons)):
        G.add_node(idx)

    EDGE_EPS = 1e-4  # siet hon 1e-3
    n = len(polygons)

    for i in range(n):
        pi = polygons[i]
        coords_i = list(pi.exterior.coords)
        edges_i = list(zip(coords_i, coords_i[1:]))

        for j in range(i + 1, n):
            pj = polygons[j]
            coords_j = list(pj.exterior.coords)
            edges_j = list(zip(coords_j, coords_j[1:]))

            share = False
            for e1 in edges_i:
                for e2 in edges_j:
                    if segments_share_edge(e1, e2, EDGE_EPS):
                        share = True
                        break
                if share:
                    break

            if share:
                G.add_edge(i, j)

    components = list(nx.connected_components(G))
    print("So component theo canh chung:", len(components))

    # ----- 2. CAT COMPONENT THEO KHOANG CACH CENTROID -----

    MAX_CENTROID_DIST = 80.0  # thu 50100, giam neu con dinh xa

   

    refined_components = []
    for comp in components:
        refined_components.extend(
            split_component_by_distance(polygons, comp, MAX_CENTROID_DIST)
        )

    print("So cluster sau khi cat theo dist:", len(refined_components))
    return cluster_by_vertex(polygons, refined_components, components, min_size=2, max_size=4)


   
