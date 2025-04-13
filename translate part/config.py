CLASS_CONFIG = {
    "voltage(2-terminal)": {
        "confidence_threshold": 0,
        "spice_symbol": "voltage",
        "ltspice_symbol": "voltage",
        "default_params": "DC 5V",
        "pin_offset": [(-20,0), (20,0)]
    },
    "resistor": {
        "confidence_threshold": 0,
        "spice_symbol": "res",
        "ltspice_symbol": "res",
        "default_params": "1k",
        "pin_offset": [(-20,0), (20,0)]
    },
    "capacitor": {
        "confidence_threshold": 0,
        "spice_symbol": "cap",
        "ltspice_symbol": "cap",
        "default_params": "1u",
        "pin_offset": [(-20,0), (20,0)]
    },
    "inductor": {
        "confidence_threshold": 0,
        "spice_symbol": "ind",
        "ltspice_symbol": "ind",
        "default_params": "1u",
        "pin_offset": [(-20,0), (20,0)]
    },
    "AC": {
        "confidence_threshold": 0,
        "spice_symbol": "current",
        "ltspice_symbol": "current",
        "default_params": "AC 1A",
        "pin_offset": [(-20,0), (20,0)]
    },
    "relay": {
        "confidence_threshold": 0,
        "spice_symbol": "ind2",
        "ltspice_symbol": "ind2",
        "default_params": "1u",
        "pin_offset": [(-20,0), (20,0)]
    },
    "voltage(1-terminal)": {
        "confidence_threshold": 0,
        "spice_symbol": "voltage",
        "ltspice_symbol": "voltage",
        "default_params": "DC 5V",
        "pin_offset": [(-20,0), (20,0)]
    },
    "ground": {
        "confidence_threshold": 0,
        "spice_symbol": "res",
        "ltspice_symbol": "res",
        "default_params": "1k",
        "pin_offset": [(-20,0), (20,0)]
    },
    "switch": {
        "confidence_threshold": 0,
        "spice_symbol": "sw",
        "ltspice_symbol": "sw",
        "default_params": "",
        "pin_offset": [(-20,0), (20,0)]
    },
    "double-switch": {
        "confidence_threshold": 0,
        "spice_symbol": "csw",
        "ltspice_symbol": "csw",
        "default_params": "",
        "pin_offset": [(-20,0), (20,0)]
    },
    "transformer": {
        "confidence_threshold": 0,
        "spice_symbol": "ind",
        "ltspice_symbol": "ind",
        "default_params": "1u",
        "pin_offset": [(-20,0), (20,0)]
    },
    "square_wave": {
        "confidence_threshold": 0,
        "spice_symbol": "current",
        "ltspice_symbol": "current",
        "default_params": "AC 1A",
        "pin_offset": [(-20,0), (20,0)]
    },
    "out-put": {
        "confidence_threshold": 0,
        "spice_symbol": "ind",
        "ltspice_symbol": "ind",
        "default_params": "1u",
        "pin_offset": [(-20,0), (20,0)]
    },
    "noise": {
        "confidence_threshold": 0,
        "spice_symbol": "ind",
        "ltspice_symbol": "ind",
        "default_params": "1u",
        "pin_offset": [(-20,0), (20,0)]
    },
    "rheostat": {
        "confidence_threshold": 0,
        "spice_symbol": "res2",
        "ltspice_symbol": "res2",
        "default_params": "1k",
        "pin_offset": [(-20,0), (20,0)]
    },
    "Audio Out": {
        "confidence_threshold": 0,
        "spice_symbol": "ind",
        "ltspice_symbol": "ind",
        "default_params": "1u",
        "pin_offset": [(-20,0), (20,0)]
    },
    "L": {
        "confidence_threshold": 0,
        "spice_symbol": "LED",
        "ltspice_symbol": "LED",
        "default_params": "",
        "pin_offset": [(-20,0), (20,0)]
    },
    # 其他组件的配置
}

OUTPUT_CONFIG = {
    "netlist_name": "circuit.net",
    "visualization_name": "visualization.jpg",
    "processed_json_name": "processed_data.json"
}

# 新增LTspice图纸尺寸配置
LTSPICE_SHEET_SIZE = (880, 680)
IMAGE_SIZE = (1280, 720) # 输入图像尺寸 
CONNECTION_MAX_DISTANCE = 0.02 # 归一化后的连接距离阈