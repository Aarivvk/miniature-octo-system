#! /bin/sh

sudo apt install nvidia-container-runtime

IMAGE=tensor/2:local

docker build -t $IMAGE .
