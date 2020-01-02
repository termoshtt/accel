//! accel: Write GPGPU code in Rust
//! ==============================
//!

extern crate cuda_driver_sys as cuda;
extern crate cuda_runtime_sys as cudart;

pub mod device;
pub mod error;
pub mod kernel;
pub mod module;
pub mod mvec;
pub mod uvec;

pub use kernel::{Block, Grid};
pub use mvec::MVec;
pub use uvec::UVec;
