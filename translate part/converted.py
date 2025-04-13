import json
import os
import glob

def convert_json_format(new_json_data):
    """执行格式转换的核心函数"""
    converted = {
        "version": "1.0",
        "components": [],
        "connections": []
    }
    
    for component in new_json_data:
        # 转换组件格式
        converted_component = {
            "type": component["class"],
            "confidence": component["confidence"],
            "position": {
                "x": component["center"][0],
                "y": component["center"][1]
            },
            "bbox": {
                "x1": component["bbox"][0],
                "y1": component["bbox"][1],
                "x2": component["bbox"][2],
                "y2": component["bbox"][3]
            },
            "pins": [{"x": pin[0], "y": pin[1]} for pin in component["pins"]]
        }
        converted["components"].append(converted_component)
    
    return converted

def batch_convert_json(input_dir, output_dir):
    """
    批量转换JSON文件
    :param input_dir: 包含新格式JSON的目录
    :param output_dir: 转换结果输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    for json_file in json_files:
        try:
            # 读取原始文件
            with open(json_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            # 执行格式转换
            converted_data = convert_json_format(original_data)
            
            # 保存转换结果
            output_path = os.path.join(
                output_dir, 
                os.path.basename(json_file)
            )
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, indent=2, ensure_ascii=False)
            
            print(f"成功转换: {os.path.basename(json_file)}")
            
        except Exception as e:
            print(f"转换失败 {os.path.basename(json_file)}: {str(e)}")

if __name__ == "__main__":
    # 配置路径
    input_directory = r"D:\test_results"  # 新格式JSON所在目录
    output_directory = r"D:\test_results\converted"  # 转换结果输出目录
    
    # 执行批量转换
    print("开始批量转换JSON格式...")
    batch_convert_json(input_directory, output_directory)
    print("\n转换完成！结果保存在:", output_directory)
