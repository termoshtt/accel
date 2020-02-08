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
use std::{cell::RefCell, mem::MaybeUninit, rc::Rc};

pub use cuda::CUctx_flags_enum as ContextFlag;
pub use cuda::CUlimit as Limit;

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

    /// Create a new CUDA context on this device.
    /// Be sure that returned context is not "current".
    ///
    /// ```
    /// # use accel::driver::device::*;
    /// let device = Device::nth(0).unwrap();
    /// let ctx = device.create_context_auto().unwrap(); // context is created, but not be "current"
    /// ```
    pub fn create_context(&self, flag: ContextFlag) -> Result<Box<Context>> {
        Ok(get_context_stack().borrow_mut().create(self, flag)?)
    }

    /// Create a new CUDA context on this device with default flag
    pub fn create_context_auto(&self) -> Result<Box<Context>> {
        self.create_context(ContextFlag::CU_CTX_SCHED_AUTO)
    }
}

/// Marker struct for CUDA Driver context
#[repr(C)]
#[derive(Debug)]
pub struct Context {
    context: CUctx_st,
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { cuCtxDestroy_v2(self.as_raw()) }
            .check()
            .expect("Context remove failed");
    }
}

impl Context {
    unsafe fn as_raw(&self) -> CUcontext {
        self as *const Context as *mut CUctx_st
    }

    unsafe fn into_raw(self: Box<Self>) -> CUcontext {
        Box::into_raw(self) as *mut CUctx_st
    }

    /// Cast pointer to Rust box (owned)
    unsafe fn as_box(ctx: CUcontext) -> Box<Self> {
        Box::from_raw(ctx as *mut Context)
    }

    pub fn version(&self) -> Result<u32> {
        let mut version: u32 = 0;
        unsafe { cuCtxGetApiVersion(self.as_raw(), &mut version as *mut _) }.check()?;
        Ok(version)
    }

    /// Set context to the "current" of this thread
    ///
    /// ```
    /// # use accel::driver::device::*;
    /// let device = Device::nth(0).unwrap();
    /// let ctx = device.create_context_auto().unwrap(); // context is created, but not be "current"
    /// let _ctx_gurad = ctx.set().unwrap();  // Push ctx to current thread
    ///                                       // Pop ctx when drop
    /// ```
    pub fn set(&self) -> Result<ContextGuard> {
        Ok(get_context_stack().borrow().set(self)?)
    }
}

/// RAII handler for CUDA context
pub struct ContextGuard<'ctx> {
    context: &'ctx Context,
}

impl<'ctx> Drop for ContextGuard<'ctx> {
    fn drop(&mut self) {
        let ctx = get_context_stack()
            .borrow_mut()
            .pop()
            .expect("Failed to pop context");
        unsafe {
            if ctx.into_raw() != self.context.as_raw() {
                panic!("Pop different CUDA context");
            }
        }
    }
}

/// Marker for context stack managed by CUDA runtime
pub struct ContextStack;
thread_local!(static CONTEXT_STACK: Rc<RefCell<ContextStack>> = Rc::new(RefCell::new(ContextStack)));

/// Get thread-local context stack managed by CUDA runtime
///
/// ```
/// # use accel::driver::device::*;
/// let device = Device::nth(0).unwrap();
/// let ctx = device.create_context_auto().unwrap();
/// let _ctx_gurad = ctx.set().unwrap(); // needs "current" context
///
/// let stack_size = get_context_stack()
///     .borrow()
///     .get_limit(Limit::CU_LIMIT_STACK_SIZE).unwrap();
/// ```
pub fn get_context_stack() -> Rc<RefCell<ContextStack>> {
    cuda_driver_init();
    CONTEXT_STACK.with(|rc| rc.clone())
}

impl ContextStack {
    pub fn set_limit(&mut self, limit: Limit, value: usize) -> Result<()> {
        unsafe { cuCtxSetLimit(limit, value) }.check()?;
        Ok(())
    }

    pub fn get_limit(&self, limit: Limit) -> Result<usize> {
        let mut value = 0;
        unsafe { cuCtxGetLimit(&mut value as *mut _, limit) }.check()?;
        Ok(value)
    }

    pub fn create(&mut self, device: &Device, flag: ContextFlag) -> Result<Box<Context>> {
        let context = unsafe {
            let mut context = MaybeUninit::uninit();
            cuCtxCreate_v2(context.as_mut_ptr(), flag as u32, device.device).check()?;
            context.assume_init()
        };
        if context.is_null() {
            bail!("Cannot crate a new context");
        }
        // Drop this context ref, and pop from stack
        self.pop()
    }

    /// Make context "current" on this thread
    pub fn set<'ctx>(&self, ctx: &'ctx Context) -> Result<ContextGuard<'ctx>> {
        unsafe { cuCtxPushCurrent_v2(ctx.as_raw()) }.check()?;
        Ok(ContextGuard { context: ctx })
    }

    /// Pops the current CUDA context from the current CPU thread.
    pub fn pop(&mut self) -> Result<Box<Context>> {
        let context = unsafe {
            let mut context = MaybeUninit::uninit();
            cuCtxPopCurrent_v2(context.as_mut_ptr()).check()?;
            context.assume_init()
        };
        if context.is_null() {
            bail!("No current context");
        }
        Ok(unsafe { Context::as_box(context) })
    }

    /// Pushes a context on the current CPU thread.
    pub fn push(&mut self, ctx: Box<Context>) -> Result<()> {
        unsafe { cuCtxPushCurrent_v2(ctx.into_raw()) }.check()?;
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
    fn create_set_limit() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let _st = ctx.set()?;

        get_context_stack()
            .borrow_mut()
            .set_limit(Limit::CU_LIMIT_STACK_SIZE, 128)?;
        Ok(())
    }

    #[should_panic]
    #[test]
    fn set_limit_none() {
        get_context_stack()
            .borrow_mut()
            .set_limit(Limit::CU_LIMIT_STACK_SIZE, 128)
            .unwrap();
    }

    #[test]
    fn create_get_limit() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let _st = ctx.set()?;

        let _stack_size = get_context_stack()
            .borrow()
            .get_limit(Limit::CU_LIMIT_STACK_SIZE)?;
        Ok(())
    }

    #[should_panic]
    #[test]
    fn get_limit_none() {
        let _stack_size = get_context_stack()
            .borrow()
            .get_limit(Limit::CU_LIMIT_STACK_SIZE)
            .unwrap();
    }

    #[test]
    fn set_get_limit() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let _st = ctx.set()?;

        get_context_stack()
            .borrow_mut()
            .set_limit(Limit::CU_LIMIT_STACK_SIZE, 128)?;
        let stack_size = get_context_stack()
            .borrow()
            .get_limit(Limit::CU_LIMIT_STACK_SIZE)?;
        assert_eq!(stack_size, 128);
        Ok(())
    }
}
