FROM nvidia/cuda:10.1-base
RUN apt update -y; \
		apt install -y vim tmux zsh; \
		apt install -y make
COPY test/ /acc_test/
