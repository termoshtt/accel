FROM nvidia/cuda:10.1-base
LABEL maintainer "Toshiaki Hishinuma <hishinuma.toshiaki@gmail.com>"

RUN apt-get update \
&&  apt-get install -y make cmake \
&&  apt-get install -y gcc-8-offload-nvptx nvptx-tools g++-8 gfortran-8 \
&&  apt-get install -y libopenblas-dev \
&&  apt-get install -y cuda-cublas-dev-10-0 cuda-cudart-dev-10-0 cuda-compiler-10.0 cuda-nvprof-10-1 cuda-cusolver-dev-10-0 cuda-cusparse-dev-10-0 

# Utils
RUN apt-get install -y python3 python3-yaml python3-numpy \
&&  apt-get install -y linux-tools-common strace trace-cmd valgrind gdb

RUN apt-get clean \
&&  rm -rf /var/lib/apt/lists/* \
&&  cp /usr/bin/g++-8 /usr/bin/g++ \
&&  cp /usr/bin/gcc-8 /usr/bin/gcc \
&&  cp /usr/bin/gfortran-8 /usr/bin/gfortran

ENV MONOLISH_DIR /lib/monolish
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$MONOLISH_DIR/lib

COPY test/ /test
WORKDIR /test
CMD ["make", "test"]
