//! accel: Write GPGPU code in Rust
//! ==============================
//!

extern crate cuda_sys as ffi;
#[macro_use]
extern crate procedurals;

pub mod error;
pub mod uvec;
pub mod kernel;
pub mod module;
pub mod device;

pub use uvec::UVec;
pub use kernel::{Block, Grid};
