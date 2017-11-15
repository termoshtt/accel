# (WIP) Accel: GPGPU programing in Rust

CUDA-based GPGPU framework for Rust

- Compile PTX Kernel from Rust using NVPTX backend of LLVM (demonstrated in [japaric/nvptx](https://github.com/japaric/nvptx))
- [proc-macro-attribute](https://github.com/rust-lang/rust/issues/38356)-based approach like [futures-await](https://github.com/alexcrichton/futures-await)
- Simple memory management using [Unified Memory](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)

```rust
#![feature(proc_macro)]

extern crate accel;
extern crate accel_derive;

use accel_derive::kernel;
use accel::*;

#[kernel]
pub fn add(a: *const f64, b: *const f64, c: *mut f64, n: usize) {
    if i < n {
      unsafe {
          *c[i] = a[i] + b[i];
      }
    }
}

fn main() {
    let n = 1024;
    let a = UVec::new(n).unwrap();
    let b = UVec::new(n).unwrap();
    let mut c = UVec::new(n).unwrap();

    let grid = Grid::x(64);
    let block = Block::x(64);
    add(grid, block, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n);
}
```
