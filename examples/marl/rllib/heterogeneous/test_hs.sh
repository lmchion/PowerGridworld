

#!/bin/bash
#
# Note:  Change the arguments based on your available resources.
#
# If running on VPN on MacOS, add the additional argument for train.py:
#    --node-ip-address $(ipconfig getifaddr en0)

Help()
{
   # Display Help
   echo "This script run training ."
   echo
   echo "Syntax: scriptTemplate [-c|l|h]"
   echo "options:"
   echo "c     location of last checkpoint"
   echo "l     true if run locally"
   echo "h     help"
   echo
}



while getopts c:l:h: flag

do
        case "${flag}" in
                c) checkpoint=${OPTARG}
                        ;;
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
    python -u ./examples/marl/rllib/heterogeneous/test_loop_hs.py \
        --stop-iters 1 \
        --stop-reward -0.5 \
        --num-cpus 1 \
        --num-gpus 0 \
        --local-dir $(pwd)/data/outputs/ray_results \
        --max-episode-steps 288 \
        --input-dir $(pwd)/data/inputs \
        --last-checkpoint $checkpoint  #./outputs/ray_results/PPO/PPO_002_ae10f_00000_0_framework=torch_2023-03-14_16-06-17/checkpoint_000001

else
    cd PowerGridworld
    python -u $(pwd)/examples/marl/rllib/heterogeneous/test_loop_hs.py \
        --stop-iters 1 \
        --stop-reward -0.5 \
        --num-cpus 4 \
        --num-gpus 1 \
        --local-dir $(pwd)/data/outputs/ray_results \
        --max-episode-steps 288 \
        --input-dir $(pwd)/data/inputs \
        --last-checkpoint $checkpoint  #./outputs/ray_results/PPO/PPO_002_ae10f_00000_0_framework=torch_2023-03-14_16-06-17/checkpoint_000001

fi