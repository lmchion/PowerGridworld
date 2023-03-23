

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
    python -u ./examples/marl/rllib/heterogeneous/train_hs.py \
        --stop-iters 200 \
        --stop-reward -0.5 \
        --num-cpus 4 \
        --num-gpus 0 \
        --local-dir ./data/outputs/ray_results \
        --max-episode-steps 288 \
        --input-dir ./data/inputs \
        
else

    python -u /PowerGridworld/examples/marl/rllib/heterogeneous/train_hs.py \
        --stop-iters 1 \
        --stop-reward -1 \
        --num-cpus 32 \
        --num-gpus 2 \
        --local-dir ./PowerGridworld/data/outputs/ray_results \
        --max-episode-steps 288 \
        --input-dir ./PowerGridworld/data/inputs

fi

