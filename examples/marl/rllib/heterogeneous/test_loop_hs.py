
import csv
import json
import os
import os.path as osp
import re
import subprocess
import time

from gridworld.log import logger

import api


def main(**args):
    directory = os.path.dirname(os.path.realpath(__file__))
    

    with open(args["input_dir"]+ '/map.json', 'r') as f:
        map = json.load(f)

    iters={}

    for num,env in enumerate(list(map.keys())):
        iters[num]=env
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
                                        '--last-checkpoint',args["last_checkpoint"],
                                        '--scenario-id',env,
                                        ],
                                        cwd=directory, capture_output=True)
        print(proc)
        timeDelta = time.time() - timeStarted                     # Get execution time.
        logger.info("Finished "+env+" process in "+str(timeDelta)+" seconds.") 
        
        # output_dir='/'.join(args["last_checkpoint"].split('/')[:-2])

        # last_checkpoint = re.search('local_path=(.*)\)\\n', str(proc.stdout, 'UTF-8') )
        # last_checkpoint=last_checkpoint.group(1)

        # with open(output_dir+'/current_iteration.json', 'w') as f:
        #     json.dump(iters, f)

        logger.info("Uploading data...") 
        current_result_dir = osp.join(args["local_dir"], "PPO")
        current_result_subdir = os.listdir(current_result_dir)

        all_runout_dirs = [osp.join(current_result_dir, d) for d in current_result_subdir if osp.isdir(osp.join(current_result_dir, d))]

        latest_subdir = max(all_runout_dirs, key=os.path.getmtime)

        api.push_data(latest_subdir, "final_validation")

        
if __name__ == "__main__":

    from args_hs import parser

    args = parser.parse_args()

    _ = main(**vars(args))