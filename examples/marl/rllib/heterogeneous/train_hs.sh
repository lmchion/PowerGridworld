

#!/bin/bash
#
# Note:  Change the arguments based on your available resources.
#
# If running on VPN on MacOS, add the additional argument for train.py:
#    --node-ip-address $(ipconfig getifaddr en0)
# TRAIN LOCALLY -> nohup ./examples/marl/rllib/heterogeneous/train_hs.sh   -l true  > train_loc_log.out &
Help()
{
   # Display Help
   echo "This script run training ."
   echo
   echo "Syntax: scriptTemplate [-l|h]"
   echo "options:"
   echo "l     true if run locally"
   echo "h     help"
   echo
}



while getopts l:h: flag

do
        case "${flag}" in
                l) run_locally=${OPTARG}
                        ;;
                h) Help
                        ;;
                *) echo "Invalid option: -$flag" 
                   Help
                ;;
        esac
done

clear

echo "running locally" $run_locally

if [ "$run_locally"  = "true" ]
then
    #cd /home/rakesh/mids/210/PowerGridworld
    #rm -rf $(pwd)/data/outputs/ray_results/PPO/*
    python3 -u $(pwd)/examples/marl/rllib/heterogeneous/train_loop_hs.py \
        --stop-iters  100 \
        --stop-reward  0.0 \
        --num-cpus 16 \
        --num-gpus 1 \
        --local-dir $(pwd)/data/outputs/ray_results \
        --max-episode-steps 288 \
        --input-dir $(pwd)/data/inputs \
        --training-iteration 100 \
        --log-level 'INFO' \
        --standalone-train true
        #--scenario-id '010' \
        
        
else
    cd /PowerGridworld
    python3 -u $(pwd)/examples/marl/rllib/heterogeneous/train_loop_hs.py \
        --stop-iters 150 \
        --stop-reward -0.5 \
        --num-cpus 1 \
        --num-gpus 0 \
        --local-dir $(pwd)/data/outputs/ray_results \
        --max-episode-steps 288 \
        --input-dir $(pwd)/data/inputs \
        --training-iteration 50 \
        --standalone-train true

fi

