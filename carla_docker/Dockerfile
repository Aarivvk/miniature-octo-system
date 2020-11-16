FROM ubuntu:18.04
#FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y apt-utils sudo python python-pip software-properties-common nano gedit

RUN pip install pygame numpy simple-pid

# install CARLA-simulator, ROS-melodic-desktop, carla-ros-bridge-melodic

RUN sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9 && \
    sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main"

RUN apt-get update

RUN apt-get install -y carla-simulator 

# RUN apt-get install -y carla-ros-bridge-melodic

ENV USER carla
ENV HOME /home/${USER}
ENV SHELL /bin/bash

RUN sh -c 'echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers'

RUN useradd -m ${USER} && gpasswd -a ${USER} sudo && echo "${USER}:${USER}" | chpasswd

USER ${USER}
WORKDIR ${HOME}
SHELL ["/bin/bash", "-c"]

RUN cp /usr/share/base-files/dot.bashrc ${HOME}/.bashrc && \
    sh -c 'echo "export PYTHONPATH=/opt/ros/melodic/lib/python2.7/dist-packages:/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.9-py2.7-linux-x86_64.egg" >> ${HOME}/.bashrc'

WORKDIR /opt/carla-simulator/bin

ENTRYPOINT ["./CarlaUE4.sh",""]
