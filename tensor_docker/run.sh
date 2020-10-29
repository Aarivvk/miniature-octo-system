IMAGE=tensor/2:local

docker run -u tf \
 --runtime=nvidia \
 --name vktensor \
 -it --network=host \
  --rm -v $HOME/dev:/home/tf/dev $IMAGE bash
