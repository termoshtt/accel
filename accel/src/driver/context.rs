//! Low-level API for [primary context] and (general) [context] management
//!
//! - The [primary context] is unique per device and shared with the CUDA runtime API.
//!   These functions allow integration with other libraries using CUDA
//!
//! [primary context]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html
//! [context]:         https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html

use super::device::Device;
use cuda::*;

#[derive(Debug, PartialEq, PartialOrd)]
pub struct Context<'device> {
    context: CUcontext,
    device: &'device Device,
}
