

#!/bin/bash
#
# Note:  Change the arguments based on your available resources.
#
# If running on VPN on MacOS, add the additional argument for train.py:
#    --node-ip-address $(ipconfig getifaddr en0)
clear

run_locally=true

echo "running locally" $run_locally

if $run_locally 
then
    python -u ./examples/marl/rllib/heterogeneous/test_hs.py \
        --stop-iters 1 \
        --stop-reward -1 \
        --num-cpus 4 \
        --num-gpus 0 \
        --local-dir ./data/outputs/ray_results \
        --max-episode-steps 288 \
        --input-dir ./data/inputs \
        --last-checkpoint ./outputs/ray_results/PPO/PPO_002_ae10f_00000_0_framework=torch_2023-03-14_16-06-17/checkpoint_000001
else

    python -u /PowerGridworld/examples/marl/rllib/heterogeneous/test_hs.py \
        --stop-iters 1 \
        --stop-reward -1 \
        --num-cpus 4 \
        --num-gpus 0 \
        --local-dir ./PowerGridworld/data/outputs/ray_results \
        --max-episode-steps 288 \
        --input-dir ./PowerGridworld/data/inputs \
        --last-checkpoint ./outputs/ray_results/PPO/PPO_002_ae10f_00000_0_framework=torch_2023-03-14_16-06-17/checkpoint_000001

fi