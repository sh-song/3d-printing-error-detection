FROM ubuntu:18.04
ARG DEBIAN_FRONTEND=noninteractive
RUN \
	apt update && \
	apt install -y python3.6 && \
	apt install -y python3-pip && \
	pip3 install numpy==1.14 && \
	pip3 install opencv-contrib-python==3.4.1.15 && \
	pip3 install matplotlib==2.2.2 && \
	apt install -y python3-tk && \
	apt install -y libsm6 libxext6 libxrender-dev && \
	pip3 install imutils && \
	pip3 install scikit-image

RUN \
	mkdir home/workspace

EXPOSE 1234
