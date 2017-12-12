//! accel: Write GPGPU code in Rust
//! ==============================
//!

#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate toml;
#[macro_use]
extern crate procedurals;
extern crate glob;
extern crate flate2;
extern crate cuda_sys as ffi;
extern crate tempdir;

pub mod error;
pub mod ptx_builder;
pub mod uvec;
pub mod kernel;
pub mod module;
pub mod device;

pub use uvec::UVec;
pub use kernel::{Grid, Block};
