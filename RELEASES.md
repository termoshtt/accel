Version 0.3.0-alpha.2 (2020-04-06)
===================================

- Minimum Supported Rust version to be 1.42

Without CUDA Runtime API
-------------------------
- Rewrite using [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
- Explicit RAII handling of CUDA Context https://gitlab.com/termoshtt/accel/-/merge_requests/51
- CUDA Managed memories
  - Device memory https://gitlab.com/termoshtt/accel/-/merge_requests/40
  - Page-locked host memory https://gitlab.com/termoshtt/accel/-/merge_requests/47
- CUDA Stream / Event handlers https://gitlab.com/termoshtt/accel/-/merge_requests/52
    - Asynchronous Kernel launch

alloc for device code
---------------------
- Global allocator using CUDA's malloc/free https://gitlab.com/termoshtt/accel/-/merge_requests/26
- `println!`, `assert_eq!` support https://gitlab.com/termoshtt/accel/-/merge_requests/25

Move to GitLab
---------------
- GitHub Actions has several problems
  - https://github.com/rust-accel/docker-action
  - https://github.com/rust-accel/container
- GPU hosted runner for GitLab CI is now working on an instance managed by RICOS Co. Ltd. https://gitlab.com/termoshtt/accel/-/merge_requests/28

Version 0.3.0-alpha.1 (2020-01-12)
===================================

[Restart Accel Project!](https://github.com/rust-accel/accel/issues/64)

Stable Rust
-------------
Stabilize Host-side code, though device-side code still requires nightly.

- Rust 2018 edition https://github.com/rust-accel/accel/pull/70
- proc-macro has been stabilized as https://github.com/rust-accel/accel/pull/63
- cargo check runs on stable Rust https://github.com/rust-accel/accel/pull/66

Update dependencies
-----------------------
- syn, quote, proc-macro2 1.0 https://github.com/rust-accel/accel/pull/67
- rust-cuda/cuda-{runtime,driver}-sys 0.3.0-alpha.1 https://github.com/rust-accel/accel/pull/66

rust-ptx-linker
-----------------
Linker flavor using rust-ptx-linker has been merged into rustc https://github.com/rust-lang/rust/pull/57937

- Rewrite accel-derive with rust-ptx-linker https://github.com/rust-accel/accel/pull/71
- archive [nvptx](https://github.com/rust-accel/nvptx) and other crates
