

import os,re
import subprocess
from itertools import permutations,cycle, islice
import random, json
import time




def main(**args):
    directory = os.path.dirname(os.path.realpath(__file__))
    

    with open(args["input_dir"]+ '/map.json', 'r') as f:
        map = json.load(f)


    perm = list(permutations( list(map.keys()), len(list(map.keys()))   ))
    random.shuffle(perm)
 
    perm = list(islice(cycle(list(perm)), args["stop_iters"]))

    last_checkpoint=None

    for env_set in perm:
        print("env_set",env_set)
        for env in env_set:
            print("env",env)
            timeStarted = time.time()  
            proc = subprocess.run(['python','-u','train_hs.py', 
                                         '--stop-iters',str(args["stop_iters"]), 
                                         '--stop-reward',str(args["stop_reward"]), 
                                         '--num-cpus',str(args["num_cpus"]), 
                                         '--num-gpus',str(args["num_gpus"]), \
                                         '--local-dir',args["local_dir"],
                                         '--max-episode-steps',str(args['max_episode_steps']),
                                         '--input-dir',args['input_dir'],
                                         '--input-file-name',str(env)+'.json',
                                         '--last-checkpoint',str(last_checkpoint),
                                         ],
                                            cwd=directory, capture_output=True)
            #print(proc)
            timeDelta = time.time() - timeStarted                     # Get execution time.
            print("Finished "+env+" process in "+str(timeDelta)+" seconds.") 

            if last_checkpoint!=None:
                del_dir = subprocess.run(['rm','-rf','/'.join(last_checkpoint.split('/')[:-1]) ])
                #print(del_dir)
                del_dir = subprocess.run(['rm','-rf','/'.join(last_checkpoint.split('/')[:-2])+'/basic-variant-state-'+run_date+'.json'])
                #print(del_dir)
                del_dir = subprocess.run(['rm','-rf','/'.join(last_checkpoint.split('/')[:-2])+'/experiment_state-'+run_date+'.json'])
                #print(del_dir)


            last_checkpoint = re.search('local_path=(.*)\)\\n', str(proc.stdout, 'UTF-8') )
            last_checkpoint=last_checkpoint.group(1)
            run_date = re.search('=torch_(.*)/checkpoint', last_checkpoint )
            run_date=run_date.group(1)

            print('last_checkpoint',last_checkpoint)



        
if __name__ == "__main__":

    from args_hs import parser

    args = parser.parse_args()

    _ = main(**vars(args))