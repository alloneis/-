import json

with open('component_map.json', 'r+') as f:
    data = json.load(f)
    data['V']['footprint'] = ''
    f.seek(0)
    json.dump(data, f, indent=2)
    f.truncate()