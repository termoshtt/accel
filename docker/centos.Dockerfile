FROM nvidia/cuda:CUDA_VERSION-devel-centosCENTOS_VERSION

COPY cuda.conf /etc/ld.so.conf.d
RUN ldconfig
ENV LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH /root/.cargo/bin:$PATH

RUN cargo install ptx-linker
RUN rustup toolchain add nightly-2020-01-02 \
 && rustup target add nvptx64-nvidia-cuda --toolchain nightly-2020-01-02
