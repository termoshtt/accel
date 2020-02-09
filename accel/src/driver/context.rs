//! Low-level API for CUDA [context].
//!
//! [context]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html

use crate::{error::*, ffi_new};
use anyhow::{bail, ensure, Result};
use cuda::*;
use std::{cell::RefCell, rc::Rc};

pub use cuda::CUctx_flags_enum as ContextFlag;

/// Marker struct for CUDA Driver context
#[derive(Debug, PartialEq)]
pub struct Context {
    active: bool,
    ptr: CUcontext,
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { cuCtxDestroy_v2(self.ptr) }
            .check()
            .expect("Context remove failed");
    }
}

thread_local! {static CONTEXT_STACK_LOCK: Rc<RefCell<Option<CUcontext>>> = Rc::new(RefCell::new(None)) }
fn get_lock() -> Rc<RefCell<Option<CUcontext>>> {
    CONTEXT_STACK_LOCK.with(|rc| rc.clone())
}

impl Context {
    pub fn create(device: CUdevice, flag: ContextFlag) -> Result<Self> {
        let ptr = ffi_new!(cuCtxCreate_v2, flag as u32, device);
        if ptr.is_null() {
            bail!("Cannot crate a new context");
        }
        CONTEXT_STACK_LOCK.with(|rc| *rc.borrow_mut() = Some(ptr));
        Ok(Context { active: true, ptr })
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn version(&self) -> Result<u32> {
        let mut version: u32 = 0;
        unsafe { cuCtxGetApiVersion(self.ptr, &mut version as *mut _) }.check()?;
        Ok(version)
    }

    /// Push to the context stack of this thread
    pub fn push(&mut self) -> Result<()> {
        let lock = get_lock();
        ensure!(lock.borrow().is_none(), "No context before push");
        unsafe { cuCtxPushCurrent_v2(self.ptr) }.check()?;
        *lock.borrow_mut() = Some(self.ptr);
        self.active = true;
        Ok(())
    }

    /// Pop from the context stack of this thread
    pub fn pop(&mut self) -> Result<()> {
        let lock = get_lock();
        if lock.borrow().is_none() {
            bail!("No countext has been set");
        }
        let ptr = ffi_new!(cuCtxPopCurrent_v2);
        if ptr.is_null() {
            bail!("No current context");
        }
        ensure!(ptr == self.ptr, "Pop must return same pointer");
        *lock.borrow_mut() = None;
        self.active = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::device::*;

    #[test]
    fn create() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        dbg!(&ctx);
        Ok(())
    }

    #[test]
    fn push() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let mut ctx = device.create_context_auto()?;
        assert!(ctx.is_active());
        assert!(ctx.push().is_err());
        Ok(())
    }

    #[test]
    fn pop() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let mut ctx = device.create_context_auto()?;
        assert!(ctx.is_active());
        ctx.pop()?;
        assert!(!ctx.is_active());
        Ok(())
    }

    #[test]
    fn push_pop() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let mut ctx = device.create_context_auto()?;
        assert!(ctx.is_active());
        ctx.pop()?;
        assert!(!ctx.is_active());
        ctx.push()?;
        assert!(ctx.is_active());
        Ok(())
    }
}
