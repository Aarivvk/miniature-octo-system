FROM ubuntu:18.04
#FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y apt-utils sudo python python-pip software-properties-common nano gedit

RUN pip install pygame numpy simple-pid

# install CARLA-simulator, ROS-melodic-desktop, carla-ros-bridge-melodic

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 92635A407F7A020C && \
    add-apt-repository "deb [arch=amd64 trusted=yes] http://dist.carla.org/carla-0.9.9/ all main" && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 81061A1A042F527D && \
    add-apt-repository "deb [arch=amd64 trusted=yes] http://dist.carla.org/carla-ros-bridge-melodic/ bionic main" && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update && apt-get install -y ros-melodic-desktop

RUN apt-get install -y carla-simulator 

RUN apt-get install -y carla-ros-bridge-melodic

ENV USER carla
ENV HOME /home/${USER}
ENV SHELL /bin/bash

RUN sh -c 'echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers'

RUN useradd -m ${USER} && gpasswd -a ${USER} sudo && echo "${USER}:${USER}" | chpasswd

USER ${USER}
WORKDIR ${HOME}
SHELL ["/bin/bash", "-c"]

RUN cp /usr/share/base-files/dot.bashrc ${HOME}/.bashrc && \
    sh -c 'echo "export PYTHONPATH=/opt/ros/melodic/lib/python2.7/dist-packages:/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.9-py2.7-linux-x86_64.egg" >> ${HOME}/.bashrc' && \
    sh -c 'echo "source /opt/ros/melodic/setup.bash" >> ${HOME}/.bashrc' && \
    sh -c 'echo "source /opt/carla-ros-bridge/melodic/setup.bash" >> ${HOME}/.bashrc'

WORKDIR /opt/carla-simulator/bin

ENTRYPOINT ["./CarlaUE4.sh",""]
