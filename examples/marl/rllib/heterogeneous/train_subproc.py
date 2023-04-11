import os
import subprocess
import api

from pathlib import Path
# nohup ./examples/marl/rllib/heterogeneous/train_subproc.py  > train_subproc_log.out &

directory= Path(__file__).parents[0]
gw_path=Path(__file__).parents[4]
print(directory)
#last_checkpoint='/home/ec2-user/PowerGridworld/data/outputs/ray_results/PPO/PPO_001_f2924_00000_0_framework=torch_2023-04-10_17-55-58'
last_checkpoint=None
proc = subprocess.run(['python','-u','train_hs.py','--stop-iters','100','--stop-reward','0.0',
                             '--num-cpus','16','--num-gpus','1','--local-dir',str(gw_path)+'/data/outputs/ray_results',
                             '--max-episode-steps','288','--input-dir', str(gw_path)+'/data/inputs','--training-iteration','100',
                             '--log-level','INFO','--scenario-id','010'], #,'--last-checkpoint',last_checkpoint],
                            cwd=str(directory), capture_output=True)

print("The exit code was: %d" % proc.returncode)
out_lines = proc.stdout.splitlines()
err_lines = proc.stderr.splitlines()
print("\n\nOUT ::: ")
print(*out_lines, sep='\n')
print("\n\nERROR ::: ")
print(*err_lines, sep='\n')

#if True:
    # print(proc.stdout)
    # run_dir=(str(proc.stdout, 'UTF-8')).strip()
    # print(run_dir)
    # run_dir=Path(run_dir).parents[0]
    #api.push_data(run_dir, "final_validation")
    #api.push_data(last_checkpoint, "final_validation")
