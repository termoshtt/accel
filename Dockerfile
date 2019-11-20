FROM nvidia/cuda:10.1-base
RUN apt update \
&&  apt install -y vim make \
&&  apt install -y gcc-8-offload-nvptx nvptx-tools g++-8 \
&&  cp /usr/bin/g++-8 /usr/bin/g++ \
&&  cp /usr/bin/gcc-8 /usr/bin/gcc \
&&  apt install -y cuda-cublas-dev-10-0 cuda-cudart-dev-10-0 cuda-compiler-10.0 \
&&  apt-get clean \
&&  rm -rf /var/lib/apt/lists/* 
