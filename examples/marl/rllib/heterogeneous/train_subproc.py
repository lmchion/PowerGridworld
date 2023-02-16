import subprocess, os

directory = os.path.dirname(os.path.realpath(__file__))

print(directory)

list_files = subprocess.run(['python','-u','train.py',
                             '--num-cpus','2','--num-gpus','0','--local-dir','./ray_results','--max-episode-steps','250'], cwd=directory)

print("The exit code was: %d" % list_files.returncode)