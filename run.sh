

# cd lab2
# echo "source poetry environment"
# source $(poetry env info --path)/bin/activate
# rm -rf model_pipeline.pkl

# echo "train cal housing model"
# python3 ./trainer/train.py

mkdir inputs
aws s3 cp s3://home-steward-s3bucket /inputs --recursive


# echo "copy model to src dir"
# mv model_pipeline.pkl src
git clone https://github.com/lmchion/PowerGridworld.git

echo "building docker container"
docker build . -t homesteward:latest

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
