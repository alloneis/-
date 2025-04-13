import json
import os
import argparse
import glob
from config import CLASS_CONFIG, IMAGE_SIZE, CONNECTION_MAX_DISTANCE 
from data_processor import normalize_coordinates, validate_components
from connection_detector import detect_connections
from netlist_generator import generate_spice_netlist

def process_single_file(input_json_path, output_base_dir):
    """处理单个JSON文件"""
    try:
        # 创建对应编号的输出目录
        file_id = os.path.splitext(os.path.basename(input_json_path))[0]
        output_dir = os.path.join(output_base_dir, file_id)
        os.makedirs(output_dir, exist_ok=True)

        # 读取输入数据
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 数据处理流程
        processed_data = normalize_coordinates(data, IMAGE_SIZE)
        processed_data = validate_components(processed_data, CLASS_CONFIG)
        processed_data = detect_connections(processed_data, CONNECTION_MAX_DISTANCE)

        # 生成网表
        spice_netlist = generate_spice_netlist(processed_data, CLASS_CONFIG)
        netlist_path = os.path.join(output_dir, "circuit.asc")
        with open(netlist_path, 'w', encoding='utf-8') as f:
            f.write(spice_netlist)

        # 保存处理后的JSON
        processed_json_path = os.path.join(output_dir, "processed_data.json")
        with open(processed_json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

        print(f"文件 {file_id} 处理完成")
        print(f"生成文件位置: {output_dir}\n")

    except Exception as e:
        print(f"处理文件 {input_json_path} 时出错: {str(e)}")
        # 如果需要可以记录详细错误日志
        # import traceback
        # traceback.print_exc()

def main():
    # 配置路径
    input_dir = r"D:\test_results\converted"
    output_base_dir = r"D:\test_results\processed_output"

    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"在 {input_dir} 目录中未找到JSON文件")
        return

    print(f"开始批量处理 {len(json_files)} 个文件...\n")
    
    for idx, json_file in enumerate(json_files, 1):
        print(f"正在处理文件 ({idx}/{len(json_files)})：{os.path.basename(json_file)}")
        process_single_file(json_file, output_base_dir)

    print("\n批量处理完成！所有结果保存在：", output_base_dir)

if __name__ == "__main__":
    main()
