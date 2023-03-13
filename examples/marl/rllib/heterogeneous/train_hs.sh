#!/bin/bash
#
# Note:  Change the arguments based on your available resources.
#
# If running on VPN on MacOS, add the additional argument for train.py:
#    --node-ip-address $(ipconfig getifaddr en0)
clear


cp /PowerGridworld/data/inputs/env_config.json /PowerGridworld/gridworld/scenarios/data/env_config.json
#cp ./data/inputs/env_config.json ./gridworld/scenarios/data/env_config.json

python -u /PowerGridworld/examples/marl/rllib/heterogeneous/train_hs.py \
#python -u ./examples/marl/rllib/heterogeneous/train_hs.py \
    --stop-iters 1 \
    --num-cpus 2 \
    --num-gpus 0 \
    --local-dir ./PowerGridworld/data/outputs/ray_results \
    --max-episode-steps 288