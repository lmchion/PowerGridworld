import os
import subprocess

directory = os.path.dirname(os.path.realpath(__file__))

print(directory)

list_files = subprocess.run(['python','-u','train_hs.py','--stop-iters','100','--stop-reward','-1',
                             '--num-cpus','4','--num-gpus','0','--local-dir','./data/outputs/ray_results',
                             '--max-episode-steps','288'],
                            cwd=directory)

print("The exit code was: %d" % list_files.returncode)