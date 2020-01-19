//! Low-level API for [device], [primary context], and (general) [context].
//!
//! - The [primary context] is unique per device and shared with the CUDA runtime API.
//!   These functions allow integration with other libraries using CUDA
//!
//! [device]:          https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html
//! [primary context]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html
//! [context]:         https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html

use super::cuda_driver_init;
use crate::error::*;
use anyhow::{bail, Result};
use cuda::*;
use std::{marker::PhantomData, mem::MaybeUninit};

pub use cuda::CUctx_flags_enum as ContextFlag;

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
    pub fn count() -> Result<usize> {
        cuda_driver_init();
        let mut count: i32 = 0;
        unsafe { cuDeviceGetCount(&mut count as *mut i32) }.check()?;
        Ok(count as usize)
    }

    pub fn new(id: i32) -> Result<Self> {
        cuda_driver_init();
        let device = unsafe {
            let mut device = MaybeUninit::uninit();
            cuDeviceGet(device.as_mut_ptr(), id).check()?;
            device.assume_init()
        };
        let primary_context = unsafe {
            let mut primary_context = MaybeUninit::uninit();
            cuDevicePrimaryCtxRetain(primary_context.as_mut_ptr(), device).check()?;
            primary_context.assume_init()
        };
        Ok(Device {
            device,
            primary_context,
        })
    }

    pub fn total_memory(&self) -> Result<usize> {
        let mut mem = 0;
        unsafe { cuDeviceTotalMem_v2(&mut mem as *mut _, self.device) }.check()?;
        Ok(mem)
    }

    pub fn get_name(&self) -> Result<String> {
        let mut bytes: Vec<u8> = vec![0_u8; 1024];
        unsafe { cuDeviceGetName(bytes.as_mut_ptr() as *mut i8, 1024, self.device) }.check()?;
        Ok(String::from_utf8(bytes)?)
    }

    /// Create a new context on the device
    pub fn create_context(&self, flags: ContextFlag) -> Result<Box<Context>> {
        cuda_driver_init();
        Ok(unsafe {
            let mut context = MaybeUninit::uninit();
            cuCtxCreate_v2(context.as_mut_ptr(), flags as u32, self.device).check()?;
            Box::from_raw(context.assume_init() as *mut Context)
        })
    }

    /// Create a new context on the device
    pub fn create_context_auto(&self) -> Result<Box<Context>> {
        self.create_context(ContextFlag::CU_CTX_SCHED_AUTO)
    }
}

// Be sure that this struct is zero-sized
#[repr(C)]
#[derive(Debug)]
pub struct Context<'device> {
    context: CUctx_st,
    phantom: PhantomData<&'device Device>,
}

impl<'device> Drop for Context<'device> {
    fn drop(&mut self) {
        unsafe { cuCtxDestroy_v2(&mut self.context as *mut _) }
            .check()
            .expect("Context remove failed");
    }
}

impl<'device> Context<'device> {
    pub fn api_version(&self) -> Result<u32> {
        let mut version: u32 = 0;
        unsafe { cuCtxGetApiVersion(&self.context as *const _ as *mut _, &mut version as *mut _) }
            .check()?;
        Ok(version)
    }

    /// Get current context with arbitary lifetime
    ///
    /// - This function returns error when no current context exists.
    ///
    pub fn get_current() -> Result<&'device Self> {
        cuda_driver_init();
        let context = unsafe {
            let mut context = MaybeUninit::uninit();
            cuCtxGetCurrent(context.as_mut_ptr()).check()?;
            context.assume_init()
        };
        if context.is_null() {
            bail!("No current context");
        }
        Ok(unsafe { (context as *mut Self).as_ref() }.unwrap())
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

    #[test]
    fn context_create() {
        let device = Device::new(0).unwrap();
        let _ctx = device.create_context_auto().unwrap();
    }

    #[test]
    fn get_current_context() {
        let device = Device::new(0).unwrap();
        let _ctx1 = device.create_context_auto().unwrap();
        let _ctx2 = Context::get_current().unwrap();
    }

    #[should_panic]
    #[test]
    fn no_current_context() {
        Context::get_current().unwrap();
    }
}
