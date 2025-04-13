import numpy as np
from scipy.spatial import KDTree

def detect_connections(data, max_distance):
    """
    基于引脚位置的连接检测
    """
    # 检查data是否为字典类型
    if not isinstance(data, dict):
        raise TypeError("输入数据必须为字典类型，而不是列表或其他类型")

    # 收集所有引脚
    pins = []
    for comp_idx, comp in enumerate(data["components"]):
        for pin in comp["pins"]:
            pins.append({
                "comp_idx": comp_idx,
                "pos": (pin["x"], pin["y"])
            })
    
    # 构建KDTree
    coords = np.array([p["pos"] for p in pins])
    if len(coords) < 2:
        return data
    
    kdtree = KDTree(coords)
    
    # 查找邻近对
    pairs = kdtree.query_pairs(max_distance)
    
    # 生成连接关系
    connections = []
    for i, j in pairs:
        src = pins[i]
        dst = pins[j]
        
        if src["comp_idx"] != dst["comp_idx"]:
            connections.append({
                "from": src["comp_idx"],
                "to": dst["comp_idx"],
                "path": [src["pos"], dst["pos"]]
            })
    
    data["connections"] = connections
    return data