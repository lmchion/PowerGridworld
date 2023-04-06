
import csv
import json
import os
import os.path as osp


from gridworld.log import logger

def push_data(logdir, csvname):
    csv_file_name = osp.join(logdir, csvname+".csv")
    json_file_name = osp.join(logdir, csvname+".json")

    with open(csv_file_name, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)

        # Convert each row into a dictionary
        # and add it to file

        file = [{k:v for k,v in rows.items() if k!=''} for rows in csvReader]

        output = {'result':file}

        with open(json_file_name, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(output, indent=4))

        import requests

        # defining the api-endpoint 
        API_ENDPOINT = "http://44.214.125.207:443/result"
        headers = {"Content-Type": "application/json; charset=utf-8"}

        # sending post request and saving response as response object
        r= requests.post(url=API_ENDPOINT, headers=headers, json=output)

        # extracting response text
        response_status = r.status_code 
        response_content = r.json()
        logger.info("data push to store; status: "+str(response_status))
        logger.info("data push to store; response: "+str(response_content))
