import os
import subprocess
import api

from pathlib import Path
# nohup ./examples/marl/rllib/heterogeneous/train_subproc.py  > train_subproc_log.out &

directory= Path(__file__).parents[0]
gw_path=Path(__file__).parents[4]
print(directory)
#last_checkpoint='/home/ec2-user/PowerGridworld/data/outputs/ray_results/PPO/PPO_010_1a1e3_00000_0_framework=torch_2023-04-08_02-28-52/checkpoint_000100'
#last_checkpoint=None
list_files = subprocess.run(['python','-u','train_hs.py','--stop-iters','200','--stop-reward','0.0',
                             '--num-cpus','16','--num-gpus','1','--local-dir',str(gw_path)+'/data/outputs/ray_results',
                             '--max-episode-steps','288','--input-dir', str(gw_path)+'/data/inputs','--training-iteration','200',
                             '--log-level','INFO','--scenario-id','010'], #,'--last-checkpoint',last_checkpoint],
                            cwd=str(directory), capture_output=True)

print("The exit code was: %d" % list_files.returncode)

if False:
    print(list_files.stdout)
    run_dir=(str(list_files.stdout, 'UTF-8')).strip()
    print(run_dir)
    run_dir=Path(run_dir).parents[0]
    api.push_data(run_dir, "final_validation")
