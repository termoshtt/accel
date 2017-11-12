//! acc: Write GPGPU code in Rust
//! ==============================
//!

#[macro_use]
extern crate procedurals;
extern crate glob;
extern crate cuda_sys as ffi;

#[macro_use]
pub mod error;
pub mod ptx_builder;
pub mod uvec;
