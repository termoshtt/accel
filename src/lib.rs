//! accel: Write GPGPU code in Rust
//! ==============================
//!

#[macro_use]
extern crate procedurals;
extern crate glob;
extern crate cuda_sys as ffi;

pub mod error;
pub mod ptx_builder;
pub mod uvec;
pub mod kernel;

pub use uvec::UVec;
pub use kernel::{Grid, Block};
