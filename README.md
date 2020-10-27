# miniature-octo-system
This is to maintain the Dockerfile for carla(on AMD graphics) and tensorflow(on NVIDIA card), enable the developments for Autonomus driving out of the box and in long term I want to make framework for AD agent and platform.

![Alt text](docs/overview.png?raw=true "Overview")

## Environment
1. OS linux(POP OS!)
1. Docker
1. NVIDIA Corporation TU116 [GeForce GTX 1660 Ti] / AMD® Radeon rx 5700 xt

## How to set up the ADM and NVIDIA graphics card in linux.
Setup link AMD: https://linuxconfig.org/amd-radeon-ubuntu-20-04-driver-installation
I use POP OS, where the setup for NVIDIA is done alredy just need update and upgrade.

## How to build?

1. build docker with dockerfile for carla and tesorflow (you can use build.sh and run.sh).
1. run the run.sh for each docker to run separately in thier own terminal.


Any sugestions are welcome, I'll try to keep it up to date as possible.


### Host environment:
1. POP os 20.04 --> Host OS
1. Radeon™ RX 5700 XT Graphics | AMD --> used for Carla simulation
1. 	GeForce GTX 1660 Ti --> used for tensorflow docker
