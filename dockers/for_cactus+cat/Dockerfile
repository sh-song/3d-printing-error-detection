FROM ubuntu:18.04
ARG DEBIAN_FRONTEND=noninteractive
RUN \
	apt update && \
	apt install -y python3.6 && \
	apt install -y python3-pip && \
    pip3 install cmake && \
	pip3 install numpy==1.16.4 && \
	pip3 install matplotlib && \
    pip3 install pandas==0.25.0 && \
	pip3 install imutils && \
	pip3 install scikit-image==0.15.0 && \
    pip3 install scipy==1.3.0 && \
    pip3 install meshcut && \
	pip3 install pygcode && \
    pip3 install scikit-learn && \
    pip3 install opencv-python-headless==4.1.0.25 && \
	apt install -y python3-tk && \
	apt install -y libsm6 libxext6 libxrender-dev && \
	pip3 install Multi-Template-Matching


RUN \
	mkdir home/workspace

EXPOSE 1234
