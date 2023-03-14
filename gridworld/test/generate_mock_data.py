


import json
from collections import OrderedDict

dir_path = '/home/lmchion/mids/w210/PowerGridworld/data/inputs/'
with open(dir_path + 'env_config.json', 'r') as f:
  data = json.load(f)


map=OrderedDict()

for i in range(10):
  key_name=str(i).zfill(3)
  map[key_name]=key_name
  with open(dir_path + key_name + '.json', 'w') as f:
        json.dump(data,f)

with open(dir_path + 'map.json', 'w') as f:
        json.dump(map,f)