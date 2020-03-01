Accel: GPGPU Framework for Rust
================================

[![pipeline status](https://gitlab.com/termoshtt/accel/badges/master/pipeline.svg)](https://gitlab.com/termoshtt/accel/commits/master)

|crate       |crates.io                                                                                      |docs.rs                                                                           |GitLab Pages                                                                                                |                                           |
|:-----------|:----------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------|:------------------------------------------|
|accel       |[![Crate](http://meritbadge.herokuapp.com/accel)](https://crates.io/crates/accel)              |[![docs.rs](https://docs.rs/accel/badge.svg)](https://docs.rs/accel)              |[![cargo-doc](https://img.shields.io/badge/doc-master-blue)](https://termoshtt.gitlab.io/accel/accel)       |CUDA-based GPGPU framework                 |
|accel-core  |[![Crate](http://meritbadge.herokuapp.com/accel-core)](https://crates.io/crates/accel-core)    |[![docs.rs](https://docs.rs/accel-core/badge.svg)](https://docs.rs/accel-core)    |Work in Progress                                                                                            |Helper for writing device code             |
|accel-derive|[![Crate](http://meritbadge.herokuapp.com/accel-derive)](https://crates.io/crates/accel-derive)|[![docs.rs](https://docs.rs/accel-derive/badge.svg)](https://docs.rs/accel-derive)|[![cargo-doc](https://img.shields.io/badge/doc-master-blue)](https://termoshtt.gitlab.io/accel/accel_derive)|Procedural macro for generating kernel code|

Requirements
------------

- Install [CUDA](https://developer.nvidia.com/cuda-downloads) on your system
  - accel depends on CUDA Runtime and Device APIs through [rust-cuda/cuda-sys](https://github.com/rust-cuda/cuda-sys)
- Setup NVPTX target of Rust
  - Install `nightly-2020-01-02` toolchain with  `nvptx64-nvidia-cuda` target, and [rust-ptx-linker](https://github.com/denzp/rust-ptx-linker)
  - There is an [setup script](setup_nvptx_toolchain.sh) for them:

```
curl -sSL https://gitlab.com/termoshtt/accel/raw/master/setup_nvptx_toolchain.sh | bash
```

Or, you can use [docekr container](./docker)

Examples
--------
See [example](accel/examples) directory.

Contribution
------------
This project is developed on [GitLab](https://gitlab.com/termoshtt/accel) and mirrored to [GitHub](https://github.com/rust-accel/accel).

Licence
--------
MIT-License

Sponsors
--------
- [RICOS Co. Ltd](https://www.ricos.co.jp/)
  - GPU instances for CI and development
