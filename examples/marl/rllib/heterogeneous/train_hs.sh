#!/bin/bash
#
# Note:  Change the arguments based on your available resources.
#
# If running on VPN on MacOS, add the additional argument for train.py:
#    --node-ip-address $(ipconfig getifaddr en0)
clear

cp /PowerGridworld/data/inputs/devices_profile_hs.csv /PowerGridworld/gridworld/agents/devices/data/devices_profile_hs.csv
cp /PowerGridworld/data/inputs/grid_cost.csv /PowerGridworld/gridworld/scenarios/data/grid_cost.csv
cp /PowerGridworld/data/inputs/pv_profile_hs.csv /PowerGridworld/gridworld/agents/pv/profiles/pv_profile_hs.csv
cp /PowerGridworld/data/inputs/vehicles_hs.csv /PowerGridworld/gridworld/agents/vehicles/vehicles_hs.csv
cp /PowerGridworld/data/inputs/env_config.json /PowerGridworld/examples/marl/rllib/heterogeneous/env_config.json


python -u /PowerGridworld/examples/marl/rllib/heterogeneous/train_hs.py \
    --stop-iters 1 \
    --num-cpus 2 \
    --num-gpus 0 \
    --local-dir ./PowerGridworld/data/outputs/ray_results \
    --max-episode-steps 288