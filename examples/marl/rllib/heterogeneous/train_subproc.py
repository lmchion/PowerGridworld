import os
import subprocess

directory = os.path.dirname(os.path.realpath(__file__))

print(directory)

list_files = subprocess.run(['python','-u','train_hs.py','--stop-iters','3',
                             '--num-cpus','2','--num-gpus','0','--local-dir','./ray_results',
                             '--max-episode-steps','250'],
                            cwd=directory)

print("The exit code was: %d" % list_files.returncode)