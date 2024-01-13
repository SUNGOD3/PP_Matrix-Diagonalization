FROM ubuntu:22.04

ARG UID
ARG USERNAME

RUN useradd -u ${UID} -m ${USERNAME}

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y libomp-dev libtbb-dev
RUN apt-get install -y vim fish git sudo python3 python3-pip
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

USER root
RUN pip3 install --no-cache-dir pytest pybind11 numpy

USER ${USERNAME}
WORKDIR /home/${USERNAME}/workspace
