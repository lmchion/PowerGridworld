


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



echo "building docker container"
sudo docker build . -t homesteward:latest


#sudo docker container run -it homesteward:latest /bin/bash


#echo "run container in a detached mode"
#sudo docker run --name hscontainer -d homesteward:latest -v /data/outputs:/PowerGridworld/examples/marl/rllib/heterogeneous/ray_results/PPO

#docker exec -it mycontainer /bin/bash

#sudo docker run --name hscontainer -d homesteward:latest

#docker exec -it hscontainer /PowerGridworld/examples/marl/rllib/heterogeneous/train_hs.sh
docker run -it --name hscontainer -d homesteward:latest bash /PowerGridworld/examples/marl/rllib/heterogeneous/train_hs.sh \ 
                 -v data/outputs:/PowerGridworld/examples/marl/rllib/heterogeneous/ray_results/PPO \
                 -v data/inputs:/PowerGridworld/data/inputs

#sudo docker container run -it homesteward:latest

#docker exec homesteward:latest /PowerGridworld/examples/marl/rllib/heterogeneous/train_hs.sh

# echo "sleep for 5 seconds"
# sleep 5

# echo "Kill the running container after the tests"
docker stop hscontainer

# echo "Delete the built docker container"
docker rm hscontainer

# echo "prune images" 
# docker image prune -f
