Accel: GPGPU Framework for Rust
================================

[![pipeline status](https://gitlab.com/termoshtt/accel/badges/master/pipeline.svg)](https://gitlab.com/termoshtt/accel/commits/master)

|crate       |crates.io                                                                                      |docs.rs                                                                           |                                           |
|:-----------|:----------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:------------------------------------------|
|accel       |[![Crate](http://meritbadge.herokuapp.com/accel)](https://crates.io/crates/accel)              |[![docs.rs](https://docs.rs/accel/badge.svg)](https://docs.rs/accel)              |CUDA-based GPGPU framework                 |
|accel-core  |[![Crate](http://meritbadge.herokuapp.com/accel-core)](https://crates.io/crates/accel-core)    |[![docs.rs](https://docs.rs/accel-core/badge.svg)](https://docs.rs/accel-core)    |Helper for writing device code             |
|accel-derive|[![Crate](http://meritbadge.herokuapp.com/accel-derive)](https://crates.io/crates/accel-derive)|[![docs.rs](https://docs.rs/accel-derive/badge.svg)](https://docs.rs/accel-derive)|Procedural macro for generating kernel code|

Requirements
------------

- Install [CUDA](https://developer.nvidia.com/cuda-downloads) on your system
  - accel depends on CUDA Runtime and Device APIs through [rust-cuda/cuda-sys](https://github.com/rust-cuda/cuda-sys)
- Setup NVPTX target of Rust
  - Install `nightly-2020-01-02` toolchain with  `nvptx64-nvidia-cuda` target, and [rust-ptx-linker](https://github.com/denzp/rust-ptx-linker)

```
curl -sSL https://gitlab.com/termoshtt/accel/raw/master/setup_nvptx_toolchain.sh | bash
```

Or, you can use [docekr container](./docker)

Example
--------

```rust
use accel::*;
use accel_derive::kernel;

#[kernel]
#[dependencies("accel-core" = "0.3.0-alpha.1")]
pub unsafe fn add(a: *const f64, b: *const f64, c: *mut f64, n: usize) {
    let i = accel_core::index();
    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}

fn main() {
    let n = 32;
    let mut a = UVec::new(n).unwrap();
    let mut b = UVec::new(n).unwrap();
    let mut c = UVec::new(n).unwrap();

    for i in 0..n {
        a[i] = i as f64;
        b[i] = 2.0 * i as f64;
    }
    println!("a = {:?}", a.as_slice());
    println!("b = {:?}", b.as_slice());

    let grid = Grid::x(1);
    let block = Block::x(n as u32);
    add(grid, block, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n);

    device::sync().unwrap();
    println!("c = {:?}", c.as_slice());
}
```

Contribution
------------
This project is developed on [GitLab](https://gitlab.com/termoshtt/accel) and mirrored to [GitHub](https://github.com/rust-accel/accel).

Licence
--------
MIT-License
