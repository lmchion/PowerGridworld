import os
import subprocess

directory = os.path.dirname(os.path.realpath(__file__))

print(directory)

list_files = subprocess.run(['python','-u','train_hs.py','--stop-iters','1000','--stop-reward','-8',
                             '--num-cpus','0','--num-gpus','1','--local-dir','./ray_results',
                             '--max-episode-steps','288'],
                            cwd=directory)

print("The exit code was: %d" % list_files.returncode)