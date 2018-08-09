//! accel: Write GPGPU code in Rust
//! ==============================
//!

extern crate cuda_sys as ffi;
#[macro_use]
extern crate procedurals;

pub mod device;
pub mod error;
pub mod kernel;
pub mod module;
pub mod uvec;

pub use kernel::{Block, Grid};
pub use uvec::UVec;
