//! Low-level API for [device] and [primary context]
//!
//! - The [primary context] is unique per device and shared with the CUDA runtime API.
//!   These functions allow integration with other libraries using CUDA
//!
//! [device]:          https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html
//! [primary context]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html

use super::{context::*, cuda_driver_init};
use crate::{error::*, ffi_new};
use anyhow::Result;
use cuda::*;

/// Handler for device and its primary context
#[derive(Debug, PartialEq, PartialOrd)]
pub struct Device {
    device: CUdevice,
    primary_context: CUcontext,
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { cuDevicePrimaryCtxRelease(self.device) }
            .check()
            .expect("Failed to release primary context");
    }
}

impl Device {
    /// Get number of available GPUs
    pub fn get_count() -> Result<usize> {
        cuda_driver_init();
        let mut count: i32 = 0;
        unsafe { cuDeviceGetCount(&mut count as *mut i32) }.check()?;
        Ok(count as usize)
    }

    pub fn nth(id: i32) -> Result<Self> {
        cuda_driver_init();
        let device = ffi_new!(cuDeviceGet, id);
        let primary_context = ffi_new!(cuDevicePrimaryCtxRetain, device);
        Ok(Device {
            device,
            primary_context,
        })
    }

    /// Get total memory of GPU
    pub fn total_memory(&self) -> Result<usize> {
        let mut mem = 0;
        unsafe { cuDeviceTotalMem_v2(&mut mem as *mut _, self.device) }.check()?;
        Ok(mem)
    }

    /// Get name of GPU
    pub fn get_name(&self) -> Result<String> {
        let mut bytes: Vec<u8> = vec![0_u8; 1024];
        unsafe { cuDeviceGetName(bytes.as_mut_ptr() as *mut i8, 1024, self.device) }.check()?;
        Ok(String::from_utf8(bytes)?)
    }

    /// Create a new CUDA context on this device.
    /// Be sure that returned context is not "current".
    ///
    /// ```
    /// # use accel::driver::device::*;
    /// let device = Device::nth(0).unwrap();
    /// let ctx = device.create_context_auto().unwrap(); // context is created, but not be "current"
    /// ```
    pub fn create_context(&self, flag: ContextFlag) -> Result<Box<Context>> {
        Ok(ContextStack::get().borrow_mut().create(self.device, flag)?)
    }

    /// Create a new CUDA context on this device with default flag
    pub fn create_context_auto(&self) -> Result<Box<Context>> {
        self.create_context(ContextFlag::CU_CTX_SCHED_AUTO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_count() -> anyhow::Result<()> {
        Device::get_count()?;
        Ok(())
    }

    #[test]
    fn get_zeroth() -> anyhow::Result<()> {
        Device::nth(0)?;
        Ok(())
    }
}
