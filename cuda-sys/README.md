cuda-sys
=========

Rust binding to CUDA Driver(`libcuda.so`)/Runtime(`libcudart.so`) APIs

- This crate does not include CUDA itself. You need to install on your own.
- `$CUDA_LIBRARY_PATH` (e.g. `/opt/cuda/lib64`) will be used for `cargo:rustc-link-search`
