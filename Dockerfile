FROM nvidia/cuda:10.1-base
LABEL maintainer "Toshiaki Hishinuma <hishinuma.toshiaki@gmail.com>"

RUN apt update \
&&  apt install -y make \
&&  apt install -y gcc-8-offload-nvptx nvptx-tools g++-8 \
&&  cp /usr/bin/g++-8 /usr/bin/g++ \
&&  cp /usr/bin/gcc-8 /usr/bin/gcc


RUN  apt install -y cuda-cublas-dev-10-0 cuda-cudart-dev-10-0 cuda-compiler-10.0 cuda-nvprof-10-1

RUN apt install -y libopenblas-dev

RUN apt-get clean \
&&  rm -rf /var/lib/apt/lists/* 

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$(HOME)/lib/monolish/lib/
