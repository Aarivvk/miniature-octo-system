From tensorflow/tensorflow:latest-gpu-jupyter

ENV DEBIAN_FRONTEND=noninteractive

ENV DISPLAY=:1

RUN apt-get update && apt-get install -y apt-utils sudo software-properties-common

RUN apt-get install -y libxkbcommon-x11-0

RUN apt-get install -y libgl1-mesa-glx

RUN apt-get install -y libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev

RUN apt-get install -y qt5-default

RUN apt-get install -y xvfb ffmpeg

RUN apt-get install -y chrpath

RUN apt-get install -y libtiff5

RUN apt-get install -y python3-opencv

RUN apt-get install -y swig

RUN pip install box2d box2d-kengz

RUN pip install --upgrade pip

RUN pip install opencv-python

RUN pip install tensorlayer