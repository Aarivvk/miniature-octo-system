p=${pwd}
cd 
docker run -u 1000:1000 --gpus=all -e HOME=/project -it --network=host --rm -v /home/vk:/project -w /project vk/tensor:local bash
cd $p