//! GPGPU framework for Rust

extern crate cuda_driver_sys as cuda;
extern crate cuda_runtime_sys as cudart;

pub mod device;
pub mod driver;
pub mod error;
pub mod mvec;
pub mod uvec;

pub use driver::kernel::{Block, Grid};
pub use mvec::MVec;
pub use uvec::UVec;

use error::Check;
use std::sync::Once;

/// Initializer for CUDA Driver API
static DRIVER_API_INIT: Once = Once::new();
fn cuda_driver_init() {
    DRIVER_API_INIT.call_once(|| {
        unsafe { cuda::cuInit(0) }
            .check()
            .expect("Initialization of CUDA Driver API failed");
    })
}
