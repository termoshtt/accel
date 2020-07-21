FROM nvidia/cuda:11.0-base
LABEL maintainer "Toshiaki Hishinuma <hishinuma.toshiaki@gmail.com>"

RUN apt-get update \
&&  apt-get install -y make cmake \
&&  apt-get install -y gcc-9-offload-nvptx nvptx-tools g++-9 gfortran-9 \
&&  apt-get install -y libopenblas-openmp-dev \
&&  apt-get install -y cuda-cudart-dev-11-0 cuda-compiler-11.0 cuda-cublas-dev-10-0 cuda-cusolver-dev-10-0 cuda-cusparse-dev-10-0  \
&&  apt-get install -y cuda-nsight-systems-11-0 nsight-systems-2020.3.2

# Utils
RUN apt-get install -y python3 python3-yaml python3-numpy \
&&  apt-get install -y linux-tools-common strace trace-cmd valgrind gdb

RUN apt-get clean \
&&  rm -rf /var/lib/apt/lists/* \

ENV MONOLISH_DIR /lib/monolish/
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/lib/monolish/lib/
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib/
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-10.0/targets/x86_64-linux/lib/
ENV MONOLISH_DIR /lib/monolish/

COPY test/ /test
