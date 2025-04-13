import numpy as np
from scipy.spatial.distance import cdist

def normalize_coordinates(data, img_size):
    """
    坐标标准化和溢出处理
    """
    width, height = img_size
    for comp in data["components"]:
        # 处理bbox坐标
        comp["bbox"]["x1"] = comp["bbox"]["x1"] % width / width
        comp["bbox"]["y1"] = comp["bbox"]["y1"] % height / height
        comp["bbox"]["x2"] = comp["bbox"]["x2"] % width / width
        comp["bbox"]["y2"] = comp["bbox"]["y2"] % height / height
        
        # 处理引脚坐标
        for pin in comp["pins"]:
            pin["x"] = pin["x"] % width / width
            pin["y"] = pin["y"] % height / height
    return data

def validate_components(data, class_config):
    """
    元件分类验证和过滤
    """
    valid_comps = []
    for comp in data["components"]:
        comp_type = comp["type"]
        cfg = class_config.get(comp_type, {})
        
        # 置信度检查
        if comp["confidence"] < cfg.get("confidence_threshold"):
            continue
            
        # 几何验证
        if not validate_geometry(comp["bbox"]):
            continue
            
        valid_comps.append(comp)
        
    data["components"] = valid_comps
    return data

def validate_geometry(bbox):
    """
    基于长宽比的几何验证
    """
    w = bbox["x2"] - bbox["x1"]
    h = bbox["y2"] - bbox["y1"]
    aspect = w / (h + 1e-6)
    return 0.1 < aspect < 100  # 有效长宽比范围
