


#!/usr/bin/bash
# 
# TRAIN -> ./run.sh -i s3://home-steward-s3bucket/scen-train/inputs/ -o s3://home-steward-s3bucket/scen-train/outputs  -t true
# TEST -> ./run.sh -i s3://home-steward-s3bucket/scen-test/inputs/ -o s3://home-steward-s3bucket/scen-test/outputs  -t false -c s3://home-steward-s3bucket/scen-train/outputs/PPO_003_204cc_00000_0_framework=torch_2023-03-14_21-56-51/checkpoint_000001/

Help()
{
   # Display Help
   echo "This script run training ."
   echo
   echo "Syntax: scriptTemplate [-i|o|h]"
   echo "options:"
   echo "i     input location in s3 directory (i.e <bucket name>/base-train/inputs/)"
   echo "o     output location in s3 directory (i.e <bucket name>/base-train/outputs/)"
   echo "t     true if train, false if test"
   echo "c     location of last checkpoint"
   echo "h     help"
   echo
}



while getopts i:o:t:c:h: flag

do
        case "${flag}" in
                i) infolder=${OPTARG}
                        ;;
                o) outfolder=${OPTARG}
                         ;;
                t) istrain=${OPTARG}
                        ;;
                c) checkpoint=${OPTARG}
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

sudo rm -rf $(pwd)/data/outputs/ray_results/PPO/*

wait 10

echo "building docker container"
sudo docker build . -t homesteward:latest  # RUn Dockerfile


echo "run container in a detached mode"

if $istrain
then
    sudo docker run -v $(pwd)/data/outputs:/PowerGridworld/data/outputs --name hscontainer -it  -d homesteward:latest bash /PowerGridworld/examples/marl/rllib/heterogeneous/train_hs.sh -l false
else
    sudo docker run -v $(pwd)/data/outputs:/PowerGridworld/data/outputs --name hscontainer -it  -d homesteward:latest bash /PowerGridworld/examples/marl/rllib/heterogeneous/test_hs.sh -c $checkpoint -l false 
fi

status_code="$(docker container wait hscontainer)"

echo "Status code of Home Steward Training: ${status_code}"



#to check inside the container
#sudo docker run -v $(pwd)/data/outputs:/PowerGridworld/data/outputs --name hscontainer -it  -d homesteward:latest
#docker exec -it hscontainer /bin/bash

aws s3 cp data/outputs/ray_results/PPO $outfolder --recursive


echo "stop the running container"
docker stop hscontainer

echo "Delete the built docker container"
docker rm hscontainer

#echo "prune images" 
#docker image prune -f
