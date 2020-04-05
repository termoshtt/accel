Accel: GPGPU Framework for Rust
================================

[![pipeline status](https://gitlab.com/termoshtt/accel/badges/master/pipeline.svg)](https://gitlab.com/termoshtt/accel/commits/master)

|crate       |crates.io                                                                                      |docs.rs                                                                           |GitLab Pages                                                                                                         |                                           |
|:-----------|:----------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|:------------------------------------------|
|accel       |[![Crate](http://meritbadge.herokuapp.com/accel)](https://crates.io/crates/accel)              |[![docs.rs](https://docs.rs/accel/badge.svg)](https://docs.rs/accel)              |[![cargo-doc](https://img.shields.io/badge/doc-master-blue)](https://termoshtt.gitlab.io/accel/accel/accel)          |CUDA-based GPGPU framework                 |
|accel-core  |[![Crate](http://meritbadge.herokuapp.com/accel-core)](https://crates.io/crates/accel-core)    |[![docs.rs](https://docs.rs/accel-core/badge.svg)](https://docs.rs/accel-core)    |[![cargo-doc](https://img.shields.io/badge/doc-master-blue)](https://termoshtt.gitlab.io/accel/accel-core/accel_core)|Helper for writing device code             |
|accel-derive|[![Crate](http://meritbadge.herokuapp.com/accel-derive)](https://crates.io/crates/accel-derive)|[![docs.rs](https://docs.rs/accel-derive/badge.svg)](https://docs.rs/accel-derive)|[![cargo-doc](https://img.shields.io/badge/doc-master-blue)](https://termoshtt.gitlab.io/accel/accel/accel_derive)   |Procedural macro for generating kernel code|

Requirements
------------
![minimum supported rust version](https://img.shields.io/badge/rustc-1.42+-red.svg)

- Minimum Supported Rust Version (MSRV) is 1.42.0
- Install [CUDA](https://developer.nvidia.com/cuda-downloads) on your system
  - accel depends on CUDA Runtime and Device APIs through [rust-cuda/cuda-sys](https://github.com/rust-cuda/cuda-sys)
- Setup NVPTX target of Rust
  - Install `nightly-2020-01-02` toolchain with  `nvptx64-nvidia-cuda` target, and [rust-ptx-linker](https://github.com/denzp/rust-ptx-linker)
  - There is an [setup script](setup_nvptx_toolchain.sh) for them:

```
curl -sSL https://gitlab.com/termoshtt/accel/raw/master/setup_nvptx_toolchain.sh | bash
```

Or, you can use [docekr container](./docker)

Contribution
------------
This project is developed on [GitLab](https://gitlab.com/termoshtt/accel) and mirrored to [GitHub](https://github.com/rust-accel/accel).

Licence
--------
Dual-licensed to be compatible with the Rust project.

- Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0
- the MIT license http://opensource.org/licenses/MIT

In addition, you must refer [End User License Agreement](https://docs.nvidia.com/cuda/eula/index.html) for using CUDA.

Sponsors
--------
- [RICOS Co. Ltd](https://www.ricos.co.jp/)
  - GPU instances for CI and development
