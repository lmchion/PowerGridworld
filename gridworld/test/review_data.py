

import json
import pandas as pd

with open('/home/lmchion/mids/w210/PowerGridworld/gridworld/test/data-01041040.json') as user_file:
  parsed_json = json.load(user_file)
  
df=None

for n1,elem in enumerate(parsed_json):
    for n2,col1 in enumerate(elem['usage_data']):
       for n3,col2 in enumerate(col1):
         if type(col2)==list:
            parsed_json[n1]['usage_data'][n2][n3]=col2[0]
       
    

for elem in parsed_json:
    temp = pd.DataFrame(elem['usage_data'],columns=elem['columns'])
    temp['device_id']=elem['device_id']

    if df is None:
       df=temp
    else:
       df=df.append(temp)

df.to_csv('/home/lmchion/mids/w210/PowerGridworld/gridworld/test/data-01041040.csv')