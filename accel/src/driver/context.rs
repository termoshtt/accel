//! Low-level API for CUDA [context].
//!
//! [context]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html

use crate::{error::*, ffi_new};
use anyhow::{bail, Result};
use cuda::*;

pub use cuda::CUctx_flags_enum as ContextFlag;

/// Marker struct for CUDA Driver context
#[derive(Debug, PartialEq)]
pub struct Context {
    ptr: CUcontext,
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { cuCtxDestroy_v2(self.ptr) }
            .check()
            .expect("Context remove failed");
    }
}

impl Context {
    pub fn version(&self) -> Result<u32> {
        let mut version: u32 = 0;
        unsafe { cuCtxGetApiVersion(self.ptr, &mut version as *mut _) }.check()?;
        Ok(version)
    }

    pub fn push(self) -> Result<()> {
        unsafe {
            cuCtxPushCurrent_v2(self.ptr).check()?;
        }
        std::mem::forget(self);
        Ok(())
    }

    pub fn pop() -> Result<Self> {
        let ptr = ffi_new!(cuCtxPopCurrent_v2);
        if ptr.is_null() {
            bail!("No current context");
        }
        Ok(Context { ptr })
    }

    pub fn create(device: CUdevice, flag: ContextFlag) -> Result<Self> {
        let ptr = ffi_new!(cuCtxCreate_v2, flag as u32, device);
        if ptr.is_null() {
            bail!("Cannot crate a new context");
        }
        Self::pop()
    }
}

#[cfg(test)]
mod tests {
    use super::super::device::*;
    use super::*;

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
        dbg!(&ctx);
        ctx.push()?;
        Ok(())
    }

    #[test]
    fn pop() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        dbg!(&ctx);
        assert!(Context::pop().is_err());
        Ok(())
    }

    #[test]
    fn push_pop() -> anyhow::Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        dbg!(&ctx);
        ctx.push()?;
        let ctx = Context::pop()?;
        dbg!(&ctx);
        Ok(())
    }
}
