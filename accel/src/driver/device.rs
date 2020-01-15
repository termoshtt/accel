//! Low-level API for [device], [primary context] and (general) [context] management
//! in [CUDA Device API].
//!
//! - The [primary context] is unique per device and shared with the CUDA runtime API.
//!   These functions allow integration with other libraries using CUDA
//!
//! [CUDA Device API]: https://docs.nvidia.com/cuda/cuda-driver-api/index.html
//! [device]:          https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html
//! [primary context]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html
//! [context]:         https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html

use super::cuda_driver_init;
use crate::error::*;
use anyhow::Result;
use cuda::*;
use std::mem::MaybeUninit;

#[derive(Debug, PartialEq, PartialOrd)]
pub struct Device(CUdevice);

#[derive(Debug, PartialEq, PartialOrd)]
pub struct Context<'device> {
    context: CUcontext,
    device: &'device Device,
}

impl Device {
    pub fn count() -> Result<usize> {
        cuda_driver_init();
        let mut count: i32 = 0;
        unsafe { cuDeviceGetCount(&mut count as *mut i32) }.check()?;
        Ok(count as usize)
    }

    pub fn new(id: i32) -> Result<Self> {
        cuda_driver_init();
        let mut device: CUdevice = 0;
        unsafe { cuDeviceGet(&mut device as *mut _, id) }.check()?;
        Ok(Device(device))
    }

    pub fn total_memory(&self) -> Result<usize> {
        let mut mem = 0;
        unsafe { cuDeviceTotalMem_v2(&mut mem as *mut _, self.0) }.check()?;
        Ok(mem)
    }

    pub fn get_name(&self) -> Result<String> {
        let mut bytes: Vec<u8> = vec![0_u8; 1024];
        unsafe { cuDeviceGetName(bytes.as_mut_ptr() as *mut i8, 1024, self.0) }.check()?;
        Ok(String::from_utf8(bytes)?)
    }

    pub fn primary_context(&self) -> Result<Context> {
        let mut context = MaybeUninit::uninit();
        unsafe {
            cuDevicePrimaryCtxRetain(context.as_mut_ptr(), self.0);
            Ok(Context {
                context: context.assume_init(),
                device: self,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_count() {
        Device::count().unwrap();
    }

    #[test]
    fn get_zeroth() {
        Device::new(0).unwrap();
    }
}
