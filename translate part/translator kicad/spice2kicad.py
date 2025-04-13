import re
import json

def load_component_map(map_file='component_map.json'):
    """加载元件映射配置"""
    with open(map_file, 'r') as f:
        return json.load(f)

def spice_to_kicad_netlist(spice_file, kicad_netlist_file, map_file='component_map.json'):
    component_map = load_component_map(map_file)
    components = []
    nets = set()
    node_counter = 1
    net_nodes = {}

    with open(spice_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('*') or not line:
                continue

            # 解析元件行（支持多节点元件）
            match = re.match(r'^(\w+)(\d+)\s+(\S+)\s+(\S+)(?:\s+(\S+))?', line)
            if match:
                prefix = match.group(1).upper()
                ref_id = prefix + match.group(2)
                nodes = list(match.groups()[2:-1])
                value = match.group(5) if match.group(5) else ""

                # 获取映射信息
                mapping = component_map.get(prefix, {})
                symbol = mapping.get('symbol', 'Device:R')
                footprint = mapping.get('footprint', 'Resistor_SMD:R_0805')

                # 记录元件
                components.append(
                    f'(comp (ref {ref_id}) (value "{value}") '
                    f'(footprint "{footprint}") (symbol "{symbol}"))'
                )

                # 记录网络节点
                for node in nodes:
                    if node not in net_nodes:
                        net_nodes[node] = node_counter
                        node_counter += 1
                    nets.add(f'(node (ref {ref_id}) (pin "{node}"))')

    # 生成KiCad网络表
    with open(kicad_netlist_file, 'w') as f:
        f.write('(export (version D)\n')
        f.write('  (components\n')
        f.write('\n'.join(components))
        f.write('\n  )\n')
        f.write('  (nets\n')
        for net_name, net_code in net_nodes.items():
            f.write(f'    (net (code {net_code}) (name "{net_name}")\n')
            f.write('      ' + '\n      '.join([n for n in nets if net_name in n]) + '\n')
            f.write('    )\n')
        f.write('  )\n)')

# 示例组件映射文件 component_map.json
"""
{
  "R": {
    "symbol": "Device:R",
    "footprint": "Resistor_SMD:R_0805"
  },
  "C": {
    "symbol": "Device:C",
    "footprint": "Capacitor_SMD:C_0603"
  },
  "V": {
    "symbol": "Power:VCC",
    "footprint": "Power:VCC"
  }
}
"""
# 运行转换
spice_to_kicad_netlist(
    'D:/test_results/processed_output/circuit.net',
    'D:/test_results/kicad_netlist.net',
    map_file='D:/test_results/component_map.json'
)
