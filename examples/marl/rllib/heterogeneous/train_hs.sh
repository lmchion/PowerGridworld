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
    cp ./data/inputs/env_config.json ./gridworld/scenarios/data/env_config.json

    python -u ./examples/marl/rllib/heterogeneous/train_hs.py \
        --stop-iters 1 \
        --stop-reward -1 \
        --num-cpus 1 \
        --num-gpus 0 \
        --local-dir ./data/outputs/ray_results \
        --max-episode-steps 288 \
        --scenario-id 9 \
        --push-data-inline False
else

    cp /PowerGridworld/data/inputs/env_config.json /PowerGridworld/gridworld/scenarios/data/env_config.json

    python -u /PowerGridworld/examples/marl/rllib/heterogeneous/train_hs.py \
        --stop-iters 1 \
        --stop-reward -1 \
        --num-cpus 0 \
        --num-gpus 1 \
        --local-dir ./PowerGridworld/data/outputs/ray_results \
        --max-episode-steps 288 \
        --scenario-id 9 \
        --push-data-inline False

fi

