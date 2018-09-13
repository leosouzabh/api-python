FROM ubuntu:latest
MAINTAINER fnndsc "leosouzabh@gmail.com"

COPY "requirements.txt" "/tmp/requirements.txt"

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && apt install -y libsm6 libxext6 \
  && apt-get install -y libgtk2.0-dev \
  && apt update \
  && pip install -r /tmp/requirements.txt

RUN /bin/bash
