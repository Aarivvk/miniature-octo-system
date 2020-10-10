IMAGE=tensor/2:local

docker run -u 1000:1000 --gpus=all \
 -e HOME=/project -it --network=host \
  --rm -v $HOME/dev:/project/dev -w /project $IMAGE bash
