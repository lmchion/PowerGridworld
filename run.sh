


#!/usr/bin/bash


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

#echo "building docker container"
#docker build . -t homesteward:latest






# echo "build container in a detached mode"
# docker run --name mycontain -d -p 8000:8000 api_test:latest

# echo "sleep for 5 seconds"
# sleep 5

# echo "running tests lab 1"
# echo "testing '/hello' endpoint with ?name=Winegar"
# curl -o /dev/null -s -w "%{http_code}\n" -X GET "http://localhost:8000/hello?name=Winegar"
# curl -sb -H "Accept: application/json" "http://localhost:8000/hello?name=Winegar" 
# printf "\n"
# echo "testing '/' endpoint"
# curl -o /dev/null -s -w "%{http_code}\n" -X GET "http://localhost:8000/"
# curl -sb -H "Accept: application/json" "http://localhost:8000/" 
# printf "\n"
# echo "testing '/docs' endpoint"
# curl -o /dev/null -s -w "%{http_code}\n" -X GET "http://localhost:8000/docs"

# echo "running tests lab 2"
# echo "testing 2 data points containing actual data"
# curl --request POST --header 'Content-Type: application/json' --data '[  { "medInc" :  8.3252, "houseAge" : 41.0, "aveRooms" : 6.98412698, 
#       "aveBedrms" : 1.02380952, "population" : 322.0, "aveOccup" : 2.55555556, 
#       "latitude" : 37.88, "longitude" :-122.23 }, 
#     { "medInc" : 8.3252, "houseAge" : 41.0, "aveRooms" : 6.98412698, 
#       "aveBedrms" : 1.02380952, "population" : 322.0, "aveOccup" : 2.55555556, 
#        "latitude" : 37.88, "longitude" :-122.23} ]' localhost:8000/predict 
# printf "\n"
# echo "testing missing longitude label "
# curl --request POST --header 'Content-Type: application/json' --data '[  { "medInc" :  8.3252, "houseAge" : 41.0, "aveRooms" : 6.98412698, 
#       "aveBedrms" : 1.02380952, "population" : 322.0, "aveOccup" : 2.55555556, 
#       "latitude" : 37.88 }]' localhost:8000/predict 
# printf "\n"
# echo "testing invalid latitude and longitude"
# curl --request POST --header 'Content-Type: application/json' -o /dev/null -s -w "%{http_code}\n" --data '[  { "medInc" :  8.3252, "houseAge" : 41.0, "aveRooms" : 6.98412698, 
#       "aveBedrms" : 1.02380952, "population" : 322.0, "aveOccup" : 2.55555556, 
#       "latitude" : 100, "longitude" : -200 }]' localhost:8000/predict 

# echo "testing string parameters"
# curl --request POST --header 'Content-Type: application/json' -o /dev/null -s -w "%{http_code}\n" --data '[  { "medInc" :  "string", "houseAge" : 41.0, "aveRooms" : 6.98412698, 
#       "aveBedrms" : 1.02380952, "population" : 322.0, "aveOccup" : 2.55555556, 
#       "latitude" : 100, "longitude" : -200 }]' localhost:8000/predict 

# echo "Kill the running container after the tests"
# docker stop mycontain

# echo "Delete the built docker container"
# docker rm mycontain

# echo "prune images" 
# docker image prune -f
