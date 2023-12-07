#! /bin/bash
source credentials.sh

docker run \
 -d \
 -p ${CONTAINER_PORT}:${CONTAINER_PORT} \
 --shm-size=8g \
 --memory=60g \
 --cpus=16 \
 --user $(id -u):$(id -g) \
 --name ${CONTAINER_NAME} \
 --rm -it --init \
 -v $(dirname $(pwd)):/app \
 --gpus all \
 ${DOCKER_NAME} \
 bash