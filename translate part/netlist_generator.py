def generate_spice_netlist(data, class_config, orig_image_size=(1920, 1080)):
    """生成符合LTspice格式的网表（优化布局版本）"""
    sheet_width = 880
    sheet_height = 680
    netlist = []
    
    # 添加头信息
    netlist.append("Version 4")
    netlist.append(f"SHEET 1 {sheet_width} {sheet_height}")
    
    # 生成元件符号和连线
    component_counter = {}
    node_map = {}
    placed_positions = set()  # 记录已放置位置
    
    # 获取原始图像尺寸
    orig_width, orig_height = orig_image_size
    
    # 生成元件符号
    for idx, comp in enumerate(data["components"]):
        comp_type = comp["type"]
        cfg = class_config.get(comp_type, {})
        
        # 获取符号配置
        symbol = cfg.get("ltspice_symbol", comp_type)
        symbol_type = cfg.get("spice_symbol", "X")
        
        # 生成实例名称
        component_counter[symbol] = component_counter.get(symbol, 0) + 1
        inst_name = f"{symbol}{component_counter[symbol]}"
        
        # 坐标转换（带随机偏移防止重叠）
        base_x = int((comp["position"]["x"] / orig_width) * sheet_width)
        base_y = int((comp["position"]["y"] / orig_height) * sheet_height)
        
        # 添加随机偏移避免重叠
        x, y = self_avoiding_placement(base_x, base_y, placed_positions)
        placed_positions.add((x, y))
        
        # 确定元件方向
        rotation = determine_component_rotation(comp)
        
        # 添加符号定义
        netlist.append(f"SYMBOL {symbol_type} {x} {y} {rotation}")
        netlist.append(f"SYMATTR InstName {inst_name}")
        if "default_params" in cfg:
            netlist.append(f"SYMATTR Value {cfg['default_params']}")
        
        # 记录引脚位置（基于调整后的坐标）
        pin_offset = get_pin_offset(comp_type, rotation)
        for pin_idx, pin in enumerate(comp["pins"]):
            px = int((pin["x"] / orig_width) * sheet_width) - base_x + x
            py = int((pin["y"] / orig_height) * sheet_height) - base_y + y
            node_map[(idx, pin_idx)] = (px + pin_offset[pin_idx][0], 
                                      py + pin_offset[pin_idx][1])
    
    # 生成连线（优化路径）
    for conn in data["connections"]:
        path = optimize_connection_path(
            [(p["x"], p["y"]) for p in conn["path"]],
            orig_image_size,
            sheet_width,
            sheet_height
        )
        for i in range(len(path)-1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            netlist.append(f"WIRE {x1} {y1} {x2} {y2}")
    
    return "\n".join(netlist)

def self_avoiding_placement(base_x, base_y, existing, grid=50, max_attempts=10):
    """自动避让布局算法"""
    for attempt in range(max_attempts):
        offset_x = (attempt // 2) * grid * (-1)**attempt
        offset_y = (attempt % 2) * grid * (-1)**(attempt//2)
        new_x = base_x + offset_x
        new_y = base_y + offset_y
        
        # 检查边界
        new_x = max(20, min(860, new_x))
        new_y = max(20, min(660, new_y))
        
        if (new_x, new_y) not in existing:
            return new_x, new_y
    return base_x, base_y  # 无法找到合适位置时返回原始位置

def get_pin_offset(comp_type, rotation):
    """获取不同元件的引脚偏移量"""
    # 基本偏移配置（根据LTspice符号尺寸）
    offsets = {
        "resistor": [(-20, 0), (20, 0)],
        "capacitor": [(-15, 0), (15, 0)],
        "voltage(2-terminal)": [(-20, -10), (20, -10)],
        # 添加更多元件类型...
    }
    
    # 应用旋转
    base = offsets.get(comp_type, [(-10,0), (10,0)])
    return apply_rotation(base, rotation)

def apply_rotation(points, rotation):
    """应用旋转变换到偏移量"""
    rot_map = {
        "R0": lambda x,y: (x,y),
        "R90": lambda x,y: (-y,x),
        "R180": lambda x,y: (-x,-y),
        "R270": lambda x,y: (y,-x)
    }
    return [rot_map[rotation](x,y) for x,y in points]

def optimize_connection_path(original_path, orig_size, sheet_w, sheet_h):
    """优化连线路径生成"""
    orig_w, orig_h = orig_size
    scaled_path = []
    
    # 坐标转换
    for x, y in original_path:
        sx = int((x / orig_w) * sheet_w)
        sy = int((y / orig_h) * sheet_h)
        scaled_path.append((sx, sy))
    
    # 简单直线优化
    optimized = [scaled_path[0]]
    for p in scaled_path[1:-1]:
        # 移除冗余中间点
        if not is_colinear(optimized[-1], p, scaled_path[-1]):
            optimized.append(p)
    optimized.append(scaled_path[-1])
    
    return optimized

def is_colinear(a, b, c, tolerance=2):
    """判断三点是否共线"""
    area = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
    return area < tolerance


def determine_component_rotation(comp):
    """根据引脚位置确定元件方向"""
    if len(comp["pins"]) < 2:
        return "R0"
    
    # 计算引脚方向
    dx = comp["pins"][1]["x"] - comp["pins"][0]["x"]
    dy = comp["pins"][1]["y"] - comp["pins"][0]["y"]
    
    if abs(dx) > abs(dy):
        return "R0" if dx > 0 else "R180"
    else:
        return "R90" if dy > 0 else "R270"