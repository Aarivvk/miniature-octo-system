IMAGE=tensor/2:local

docker run -u 1000:1000 --gpus=all \
 --name vktensor \
 -it --network=host \
  --rm -v $HOME/dev:/home/tf/dev -w /home/tf $IMAGE bash
