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
    python -u ./examples/marl/rllib/heterogeneous/train_hs.py \
        --stop-iters 1 \
        --stop-reward -1 \
        --num-cpus 4 \
        --num-gpus 0 \
        --local-dir ./data/outputs/ray_results \
        --max-episode-steps 288 \
        --input-dir ./data/inputs \
        
else

    python -u /PowerGridworld/examples/marl/rllib/heterogeneous/train_hs.py \
        --stop-iters 1 \
        --stop-reward -1 \
        --num-cpus 4 \
        --num-gpus 0 \
        --local-dir ./PowerGridworld/data/outputs/ray_results \
        --max-episode-steps 288 \
        --input-dir ./PowerGridworld/data/inputs

fi

