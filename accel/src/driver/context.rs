//! Low-level API for context management based on
//! [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX)

use super::device::Device;
use cuda::*;

#[derive(Debug, PartialEq, PartialOrd)]
pub struct Context<'device> {
    context: CUcontext,
    device: &'device Device,
}
