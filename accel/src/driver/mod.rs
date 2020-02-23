//! Safe Rust binding for [CUDA Driver APIs](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html).

pub mod context;
pub mod device;
pub mod kernel;
pub mod linker;
pub mod module;

use crate::{error::Check, ffi_call_unsafe};
use std::sync::Once;

pub use device::Device;

/// Initializer for CUDA Driver API
static DRIVER_API_INIT: Once = Once::new();
fn cuda_driver_init() {
    DRIVER_API_INIT.call_once(|| {
        ffi_call_unsafe!(cuda::cuInit, 0).expect("Initialization of CUDA Driver API failed");
    })
}
