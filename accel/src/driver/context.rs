//! Low-level API for CUDA [context].
//!
//! [context]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html

use crate::{ffi_call_unsafe, ffi_new_unsafe};
use anyhow::{bail, ensure, Result};
use cuda::*;
use std::{cell::RefCell, rc::Rc};

pub use cuda::CUctx_flags_enum as ContextFlag;

/// Marker struct for CUDA Driver context
#[derive(Debug)]
pub struct Context {
    ptr: CUcontext,
}

impl Drop for Context {
    fn drop(&mut self) {
        ffi_call_unsafe!(cuCtxDestroy_v2, self.ptr).expect("Context remove failed");
    }
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

thread_local! {static CONTEXT_STACK_LOCK: Rc<RefCell<Option<CUcontext>>> = Rc::new(RefCell::new(None)) }
fn get_lock() -> Rc<RefCell<Option<CUcontext>>> {
    CONTEXT_STACK_LOCK.with(|rc| rc.clone())
}

impl Context {
    /// Create on the top of context stack
    pub fn create(device: CUdevice, flag: ContextFlag) -> Result<Self> {
        let ptr = ffi_new_unsafe!(cuCtxCreate_v2, flag as u32, device)?;
        if ptr.is_null() {
            bail!("Cannot crate a new context");
        }
        CONTEXT_STACK_LOCK.with(|rc| *rc.borrow_mut() = Some(ptr));
        Ok(Context { ptr })
    }

    /// Check this context is "current" on this thread
    pub fn is_current(&self) -> Result<bool> {
        let current = ffi_new_unsafe!(cuCtxGetCurrent)?;
        Ok(current == self.ptr)
    }

    pub fn version(&self) -> Result<u32> {
        let mut version: u32 = 0;
        ffi_call_unsafe!(cuCtxGetApiVersion, self.ptr, &mut version as *mut _)?;
        Ok(version)
    }

    /// Push to the context stack of this thread
    pub fn push(&self) -> Result<()> {
        let lock = get_lock();
        ensure!(
            lock.borrow().is_none(),
            "Context already exists on this thread. Please pop it before push new context."
        );
        ffi_call_unsafe!(cuCtxPushCurrent_v2, self.ptr)?;
        *lock.borrow_mut() = Some(self.ptr);
        Ok(())
    }

    /// Pop from the context stack of this thread
    pub fn pop(&self) -> Result<()> {
        let lock = get_lock();
        if lock.borrow().is_none() {
            bail!("No countext has been set");
        }
        let ptr = ffi_new_unsafe!(cuCtxPopCurrent_v2)?;
        if ptr.is_null() {
            bail!("No current context");
        }
        ensure!(ptr == self.ptr, "Pop must return same pointer");
        *lock.borrow_mut() = None;
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
        let ctx = device.create_context_auto()?;
        assert!(ctx.is_current()?);
        assert!(ctx.push().is_err());
        Ok(())
    }

    #[test]
    fn pop() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        assert!(ctx.is_current()?);
        ctx.pop()?;
        assert!(!ctx.is_current()?);
        Ok(())
    }

    #[test]
    fn push_pop() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        assert!(ctx.is_current()?);
        ctx.pop()?;
        assert!(!ctx.is_current()?);
        ctx.push()?;
        assert!(ctx.is_current()?);
        Ok(())
    }

    #[test]
    fn thread() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let ctx1 = device.create_context_auto()?;
        assert!(ctx1.is_current()?); // "current" on this thread
        let th = std::thread::spawn(move || -> anyhow::Result<()> {
            assert!(!ctx1.is_current()?); // ctx1 is NOT current on this thread
            let ctx2 = device.create_context_auto()?;
            assert!(ctx2.is_current()?);
            Ok(())
        });
        th.join().unwrap()?;
        Ok(())
    }
}
