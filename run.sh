


#!/usr/bin/bash
# ./run.sh -i s3://home-steward-s3bucket/base-train/inputs/ -o s3://home-steward-s3bucket/base-train/outputs

Help()
{
   # Display Help
   echo "This script run training ."
   echo
   echo "Syntax: scriptTemplate [-i|o|h]"
   echo "options:"
   echo "i     input location in s3 directory (i.e <bucket name>/base-train/inputs/)"
   echo "o     output location in s3 directory (i.e <bucket name>/base-train/outputs/)"
   echo "h     help"
   echo
}



while getopts i:o:h: flag

do
        case "${flag}" in
                i) infolder=${OPTARG}
                        ;;
                o) outfolder=${OPTARG}
                         ;;
                h) Help
                        ;;
                *) echo "Invalid option: -$flag" 
                   Help
                ;;
        esac
done

echo "building docker container"
aws s3 cp $infolder data/inputs --recursive

cp data/inputs/devices_profile_hs.csv gridworld/agents/devices/data
cp data/inputs/grid_cost.csv gridworld/scenarios/data
cp data/inputs/pv_profile_hs.csv gridworld/agents/pv/profiles
cp data/inputs/vehicles_hs.csv gridworld/agents/vehicles/vehicles_hs.csv
cp data/inputs/env_config.json examples/marl/rllib/heterogeneous/env_config.json

echo "building docker container"
sudo docker build . -t homesteward:latest

#echo "run container in a detached mode"
#sudo docker run --name hscontainer -d homesteward:latest


#docker exec -it hscontainer /PowerGridworld/examples/marl/rllib/heterogeneous/train_hs.sh

# echo "sleep for 5 seconds"
# sleep 5

# echo "Kill the running container after the tests"
# docker stop mycontain

# echo "Delete the built docker container"
# docker rm mycontain

# echo "prune images" 
# docker image prune -f
