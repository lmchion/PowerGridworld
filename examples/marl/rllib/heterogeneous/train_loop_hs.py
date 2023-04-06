

import glob
import json
import os
import random
import re
import subprocess
import time
import api

def main(**args):
    directory = os.path.dirname(os.path.realpath(__file__))
    

    with open(args["input_dir"]+ '/map.json', 'r') as f:
        map = json.load(f)


    #perm = list(permutations( list(map.keys()), len(list(map.keys()))   ))
    #random.shuffle(perm)
 
    #perm = list(islice(cycle(list(perm)), args["stop_iters"]))
    iters={}
    last_checkpoint=None
    env_set=list(map.keys())

    cnt=0
    iter_set=int(args["stop_iters"])/int(args["training_iteration"])
    for n1,i in enumerate(range(0,int(iter_set))):
        print("iteration set",i+1)
        random.shuffle(env_set)
        print("env_set",env_set)
        iters[i]=env_set
        for n2,env in enumerate(env_set):
            print("env",env)
            timeStarted = time.time()  
            cnt +=1
            training_iteration=int(int(args["training_iteration"])*cnt)
            print("training_iteration",training_iteration,args["training_iteration"],cnt )
            proc = subprocess.run(['python','-u','train_hs.py', 
                                         '--stop-iters',str(args["stop_iters"]), 
                                         '--stop-reward',str(args["stop_reward"]), 
                                         '--num-cpus',str(args["num_cpus"]), 
                                         '--num-gpus',str(args["num_gpus"]), 
                                         '--local-dir',args["local_dir"],
                                         '--max-episode-steps',str(args['max_episode_steps']),
                                         '--input-dir',args['input_dir'],
                                         '--last-checkpoint',str(last_checkpoint),
                                         '--training-iteration',str(training_iteration),
                                        '--scenario-id',env,
                                         ],
                                            cwd=directory, capture_output=True)
            out_lines = proc.stdout.splitlines()
            err_lines = proc.stderr.splitlines()
            print("\n\nOUT ::: ")
            print(*out_lines, sep='\n')
            print("\n\nERROR ::: ")
            print(*err_lines, sep='\n')

            timeDelta = time.time() - timeStarted                     # Get execution time.
            print("Finished "+env+" process in "+str(timeDelta)+" seconds.") 

            if last_checkpoint!=None:

                del_dir = subprocess.run(['rm','-rf',prior_run_dir ])
                print(del_dir)
                for f in glob.glob( output_dir+"/*"+run_date[:-2]+"*.json"):
                   os.remove(f)

                # del_dir = subprocess.run(['rm','-rf',output_dir+'/basic-variant-state-'+run_date+'.json'])
                # print(del_dir)
                # del_dir = subprocess.run(['rm','-rf',output_dir+'/experiment_state-'+run_date+'.json'])
                # print(del_dir)


            print('######################',str(proc.stdout, 'UTF-8'))
            # last_checkpoint = re.search('local_path=(.*)\)\\n', str(proc.stdout, 'UTF-8') )
            # last_checkpoint=last_checkpoint.group(1)
            last_checkpoint=str(proc.stdout, 'UTF-8')
            run_date = re.search('=torch_(.*)/checkpoint', last_checkpoint )
            run_date=run_date.group(1)
            prior_run_dir='/'.join(last_checkpoint.split('/')[:-1])
            output_dir='/'.join(last_checkpoint.split('/')[:-2])
            api.push_data(output_dir, "final_validation")

            print('last_checkpoint',last_checkpoint)

        with open(output_dir+'/current_iteration.json', 'w') as f:
            json.dump(iters, f)

        
if __name__ == "__main__":

    from args_hs import parser

    args = parser.parse_args()

    _ = main(**vars(args))