
import csv
import json
import os
import os.path as osp
import subprocess
import time

from gridworld.log import logger


def _push_data(logdir, csvname):
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

def main(**args):
    directory = os.path.dirname(os.path.realpath(__file__))
    

    with open(args["input_dir"]+ '/map.json', 'r') as f:
        map = json.load(f)



    for env in list(map.keys()):
        print("env",env)
        timeStarted = time.time()  
        proc = subprocess.run(['python','-u','test_hs.py', 
                                        '--stop-iters',str(args["stop_iters"]), 
                                        '--stop-reward',str(args["stop_reward"]), 
                                        '--num-cpus',str(args["num_cpus"]), 
                                        '--num-gpus',str(args["num_gpus"]), \
                                        '--local-dir',args["local_dir"],
                                        '--max-episode-steps',str(args['max_episode_steps']),
                                        '--input-dir',args['input_dir'],
                                        '--input-file-name',str(env)+'.json',
                                        '--last-checkpoint',args["last_checkpoint"],
                                        '--scenario-id',env,
                                        ],
                                        cwd=directory, capture_output=True)
        print(proc)
        timeDelta = time.time() - timeStarted                     # Get execution time.
        logger.info("Finished "+env+" process in "+str(timeDelta)+" seconds.") 

        logger.info("Uploading data...") 
        current_result_dir = osp.join(args["local_dir"], "PPO")
        current_result_subdir = os.listdir(current_result_dir)

        all_runout_dirs = [osp.join(current_result_dir, d) for d in current_result_subdir if osp.isdir(osp.join(current_result_dir, d))]

        latest_subdir = max(all_runout_dirs, key=os.path.getmtime)

        _push_data(latest_subdir, "final_validation")

        
if __name__ == "__main__":

    from args_hs import parser

    args = parser.parse_args()

    _ = main(**vars(args))