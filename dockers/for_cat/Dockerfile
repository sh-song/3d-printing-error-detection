FROM ubuntu:18.04
ARG DEBIAN_FRONTEND=noninteractive

RUN \
	apt update && \
	apt install -y python3.6 && \
	apt install -y python3-pip && \
	pip3 install scikit-build && \
	pip3 install cmake && \
	pip3 install matplotlib && \
	pip3 install scipy && \
	pip3 install scikit-image && \
	pip3 install meshcut && \
	pip3 install pygcode && \
	pip3 install scikit-learn && \
	pip3 install Multi-Template-Matching && \
	apt install -y python3-tk

RUN \
	mkdir home/workspace

EXPOSE 1234
