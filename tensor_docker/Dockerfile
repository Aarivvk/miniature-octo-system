FROM tensorflow/tensorflow:latest-gpu-jupyter

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

RUN pip install rospkg catkin_pkg

RUN pip install tf-agents gym

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 81061A1A042F527D && \
    add-apt-repository "deb [arch=amd64 trusted=yes] http://dist.carla.org/carla-ros-bridge-melodic/ bionic main" && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update && apt-get install -y ros-melodic-desktop python3-yaml carla-ros-bridge-melodic

RUN apt-get install -y nano gedit

RUN pip install stable-baselines3

RUN pip install imageio-ffmpeg

RUN pip install keras

ENV USER tf
ENV HOME /home/${USER}
ENV SHELL /bin/bash

RUN sh -c 'echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers'

RUN useradd -m ${USER} && gpasswd -a ${USER} sudo && echo "${USER}:${USER}" | chpasswd

USER ${USER}
WORKDIR ${HOME}
SHELL ["/bin/bash", "-c"]

RUN cp /usr/share/base-files/dot.bashrc ${HOME}/.bashrc && \
    sh -c 'echo "source /opt/ros/melodic/setup.bash" >> ${HOME}/.bashrc' && \
    sh -c 'echo "export PYTHONPATH=${HOME}/dev/carla_env/carla-0.9.9-py3.7-linux-x86_64.egg:${PYTHONPATH}" >> ${HOME}/.bashrc' && \
    sh -c 'echo "source /opt/carla-ros-bridge/melodic/setup.bash" >> ${HOME}/.bashrc'
