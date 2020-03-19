//! GPGPU framework for Rust

extern crate cuda_driver_sys as cuda;
extern crate cuda_runtime_sys as cudart;

pub mod device;
pub mod driver;
pub mod error;
pub mod mvec;
pub mod uvec;

pub use driver::{Block, Grid};
pub use mvec::MVec;
pub use uvec::UVec;
