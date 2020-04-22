#!/bin/bash
set -xue

NIGHTLY=nightly-2020-01-02
rustup toolchain add ${NIGHTLY}
rustup target add nvptx64-nvidia-cuda --toolchain ${NIGHTLY}
rustup component add rustfmt --toolchain ${NIGHTLY}
cargo install ptx-linker -f
