import subprocess

list_files = subprocess.run(['python','-u','/home/lmchion/mids/w210/PowerGridworld/examples/marl/rllib/heterogeneous/train.py','--num-cpus','2','--num-gpus','0','--local-dir','./ray_results','--max-episode-steps','250'])

print("The exit code was: %d" % list_files.returncode)