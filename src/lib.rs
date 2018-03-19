//! accel: Write GPGPU code in Rust
//! ==============================
//!

extern crate cuda_sys as ffi;
extern crate glob;
#[macro_use]
extern crate procedurals;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate tempdir;
extern crate toml;

pub mod error;
pub mod ptx_builder;
pub mod uvec;
pub mod kernel;
pub mod module;
pub mod device;

pub use uvec::UVec;
pub use kernel::{Block, Grid};
