//! Low-level API for CUDA [context].
//!
//! [context]:         https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html

use super::cuda_driver_init;
use crate::error::*;
use anyhow::{bail, Result};
use cuda::*;
use std::{cell::RefCell, mem::MaybeUninit, rc::Rc};

pub use cuda::CUctx_flags_enum as ContextFlag;

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
    pub fn create(&mut self, device: CUdevice, flag: ContextFlag) -> Result<Box<Context>> {
        let context = unsafe {
            let mut context = MaybeUninit::uninit();
            cuCtxCreate_v2(context.as_mut_ptr(), flag as u32, device).check()?;
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
mod tests {
    use super::super::device::*;

    #[test]
    fn create() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let _ctx = device.create_context_auto()?;
        Ok(())
    }
}
