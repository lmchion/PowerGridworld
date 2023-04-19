


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

sudo rm -rf data/outputs/*

wait 20

echo "building docker container"
sudo docker build . -t homesteward:latest  # RUn Dockerfile


echo "run container in a detached mode"
sudo docker run -v $(pwd)/data/outputs:/PowerGridworld/data/outputs --name hscontainer -it  -d homesteward:latest bash /PowerGridworld/examples/marl/rllib/heterogeneous/train_hs.sh 

status_code="$(docker container wait hscontainer)"

echo "Status code of Home Steward Training: ${status_code}"



#to check inside the container
#docker exec -it hscontainer /bin/bash


aws s3 cp data/outputs/ray_results/PPO $outfolder --recursive


echo "stop the running container"
docker stop hscontainer

echo "Delete the built docker container"
docker rm hscontainer

#echo "prune images" 
#docker image prune -f
