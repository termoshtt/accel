FROM nvidia/cuda:10.1-base
RUN apt update \
&&	apt install -y vim make\
&&	apt install -y gcc-8-offload-nvptx nvptx-tools g++-8
RUN ln -s /usr/bin/g++-8 /usr/bin/g++ \
&& ln -s /usr/bin/gcc-8 /usr/bin/gcc \
