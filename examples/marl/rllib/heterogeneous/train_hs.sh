#!/bin/bash
#
# Note:  Change the arguments based on your available resources.
#
# If running on VPN on MacOS, add the additional argument for train.py:
#    --node-ip-address $(ipconfig getifaddr en0)
clear

cp /PowerGridworld/data/inputs/devices_profile_hs.csv gridworld/agents/devices/data/devices_profile_hs.csv
cp /PowerGridworld/data/inputs/grid_cost.csv gridworld/scenarios/data/grid_cost.csv
cp /PowerGridworld/data/inputs/pv_profile_hs.csv gridworld/agents/pv/profiles/pv_profile_hs.csv
cp /PowerGridworld/data/inputs/vehicles_hs.csv gridworld/agents/vehicles/vehicles_hs.csv
cp /PowerGridworld/data/inputs/env_config.json examples/marl/rllib/heterogeneous/env_config.json


python -u train_hs.py \
    --stop-iters 100 \
    --num-cpus 2 \
    --num-gpus 0 \
    --local-dir ./ray_results \
    --max-episode-steps 288