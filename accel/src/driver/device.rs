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
    /// Get number of available GPUs
    pub fn get_count() -> Result<usize> {
        cuda_driver_init();
        let mut count: i32 = 0;
        unsafe { cuDeviceGetCount(&mut count as *mut i32) }.check()?;
        Ok(count as usize)
    }

    pub fn nth(id: i32) -> Result<Self> {
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

    /// Create a new context on the device
    ///
    /// If a context is already current to the thread,
    /// it is supplanted by the newly created context and may be restored by a subsequent call to cuCtxPopCurrent().
    pub fn create_context(&self, flags: ContextFlag) -> Result<Box<Context>> {
        cuda_driver_init();
        Ok(unsafe {
            let mut context = MaybeUninit::uninit();
            cuCtxCreate_v2(context.as_mut_ptr(), flags as u32, self.device).check()?;
            Box::from_raw(context.assume_init() as *mut Context)
        })
    }

    /// Create a new context on the device with defacult option (`CU_CTX_SCHED_AUTO`)
    pub fn create_context_auto(&self) -> Result<Box<Context>> {
        self.create_context(ContextFlag::CU_CTX_SCHED_AUTO)
    }
}

/// Handler for CUDA Driver context
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

    pub fn set_limit_current(limit: CUlimit, value: usize) -> Result<()> {
        unsafe { cuCtxSetLimit(limit, value) }.check()?;
        Ok(())
    }

    pub fn get_limit_current(limit: CUlimit) -> Result<usize> {
        let mut value = 0;
        unsafe { cuCtxGetLimit(&mut value as *mut _, limit) }.check()?;
        Ok(value)
    }

    /// Get current context with arbitary lifetime
    ///
    /// - This function returns error when no current context exists.
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

    /// Binds the specified CUDA context to the calling CPU thread.
    ///
    /// If there exists a CUDA context stack on the calling CPU thread, this will replace the top of that stack
    pub fn set_current(&self) -> Result<()> {
        unsafe { cuCtxSetCurrent(&self.context as *const _ as *mut _) }.check()?;
        Ok(())
    }

    /// Pops the current CUDA context from the current CPU thread.
    pub fn pop_current() -> Result<&'device Self> {
        cuda_driver_init();
        let context = unsafe {
            let mut context = MaybeUninit::uninit();
            cuCtxPopCurrent_v2(context.as_mut_ptr()).check()?;
            context.assume_init()
        };
        if context.is_null() {
            bail!("No current context");
        }
        Ok(unsafe { (context as *mut Self).as_ref() }.unwrap())
    }

    /// Pushes a context on the current CPU thread.
    pub fn push_current(&self) -> Result<()> {
        unsafe { cuCtxPushCurrent_v2(&self.context as *const _ as *mut _) }.check()?;
        Ok(())
    }
}

#[cfg(test)]
mod device_tests {
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

#[cfg(test)]
mod context_tests {
    use super::*;

    #[test]
    fn create() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let _ctx = device.create_context_auto()?;
        Ok(())
    }

    #[test]
    fn create_get() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let _ctx1 = device.create_context_auto()?;
        let _ctx2 = Context::get_current()?;
        Ok(())
    }

    #[test]
    fn create_pop() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let _ctx = device.create_context_auto()?;
        let _poped_ctx = Context::pop_current()?;
        Ok(())
    }

    #[should_panic]
    #[test]
    fn get_none() {
        Context::get_current().unwrap();
    }

    #[should_panic]
    #[test]
    fn pop_none() {
        Context::pop_current().unwrap();
    }

    #[test]
    fn create_set_limit() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let _ctx = device.create_context_auto()?;
        Context::set_limit_current(CUlimit::CU_LIMIT_STACK_SIZE, 128)?;
        Ok(())
    }

    #[should_panic]
    #[test]
    fn set_limit_none() {
        Context::set_limit_current(CUlimit::CU_LIMIT_STACK_SIZE, 128).unwrap();
    }

    #[test]
    fn create_get_limit() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let _ctx = device.create_context_auto()?;
        let _stack_size = Context::get_limit_current(CUlimit::CU_LIMIT_STACK_SIZE)?;
        Ok(())
    }

    #[should_panic]
    #[test]
    fn get_limit_none() {
        let _stack_size = Context::get_limit_current(CUlimit::CU_LIMIT_STACK_SIZE).unwrap();
    }

    #[test]
    fn set_get_limit() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let _ctx = device.create_context_auto()?;
        Context::set_limit_current(CUlimit::CU_LIMIT_STACK_SIZE, 128)?;
        let limit = Context::get_limit_current(CUlimit::CU_LIMIT_STACK_SIZE)?;
        assert_eq!(limit, 128);
        Ok(())
    }
}
