import os
import subprocess

# directory = os.path.dirname(os.path.realpath(__file__),'../../')

# print(directory)

from pathlib import Path

directory= Path(__file__).parents[0]
gw_path=Path(__file__).parents[4]
print(directory)

list_files = subprocess.run(['python','-u','train_hs.py','--stop-iters','200','--stop-reward','-0.5',
                             '--num-cpus','1','--num-gpus','1','--local-dir',str(gw_path)+'/data/outputs/ray_results',
                             '--max-episode-steps','288','--input-dir', str(gw_path)+'/data/inputs','--training-iteration','200',
                             '--log-level','INFO','--scenario-id','010'],
                            cwd=str(directory))

print("The exit code was: %d" % list_files.returncode)


