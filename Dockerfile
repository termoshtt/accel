FROM nvidia/cuda:10.1-base
RUN apt update -y; \
		apt install -y vim tmux zsh; \
		apt install -y gcc-8-offload-nvptx nvptx-tools g++-8; \
		apt install -y make; \
		alias gcc='gcc-8'; \
		alias g++='g++-8'
COPY test/ /acc_test/
