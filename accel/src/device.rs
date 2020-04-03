//! Handler for CUDA [device] and [context]
//!
//! [device]:  https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html
//! [context]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html

use super::cuda_driver_init;
use crate::{error::Result, ffi_call, ffi_new};
use cuda::*;

/// Handler for device and its primary context
#[derive(Debug, PartialEq, PartialOrd)]
pub struct Device {
    device: CUdevice,
}

impl Device {
    /// Get number of available GPUs
    pub fn get_count() -> Result<usize> {
        cuda_driver_init();
        let mut count: i32 = 0;
        ffi_call!(cuDeviceGetCount, &mut count as *mut i32)?;
        Ok(count as usize)
    }

    pub fn nth(id: i32) -> Result<Self> {
        cuda_driver_init();
        let device = ffi_new!(cuDeviceGet, id)?;
        Ok(Device { device })
    }

    /// Get total memory of GPU
    pub fn total_memory(&self) -> Result<usize> {
        let mut mem = 0;
        ffi_call!(cuDeviceTotalMem_v2, &mut mem as *mut _, self.device)?;
        Ok(mem)
    }

    /// Get name of GPU
    pub fn get_name(&self) -> Result<String> {
        let mut bytes: Vec<u8> = vec![0_u8; 1024];
        ffi_call!(
            cuDeviceGetName,
            bytes.as_mut_ptr() as *mut i8,
            1024,
            self.device
        )?;
        Ok(String::from_utf8(bytes).expect("GPU name is not UTF8"))
    }

    /// Create a new CUDA context on this device.
    /// Be sure that returned context is not "current".
    ///
    /// ```
    /// # use accel::device::*;
    /// let device = Device::nth(0).unwrap();
    /// let ctx = device.create_context(); // context is created, but not be "current"
    /// ```
    pub fn create_context(&self) -> Context {
        Context::create(self.device)
    }
}

/// RAII handler for using CUDA context
///
/// As described in [CUDA Programming Guide], library using CUDA should push context before using
/// it, and then pop it.
///
/// [CUDA Programming Guide]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#context
pub(crate) struct ContextGuard<'lock> {
    ctx: &'lock Context,
}

impl<'lock> ContextGuard<'lock> {
    /// Make context as current on this thread
    pub fn guard_context(ctx: &'lock Context) -> Self {
        ctx.push();
        Self { ctx }
    }
}

impl<'lock> Drop for ContextGuard<'lock> {
    fn drop(&mut self) {
        self.ctx.pop();
    }
}

/// Object tied up to a CUDA context
pub(crate) trait Contexted {
    fn get_context(&self) -> &Context;

    /// RAII utility for push/pop onto the thread-local context stack
    fn guard_context<'lock>(&'lock self) -> ContextGuard<'lock> {
        let ctx = self.get_context();
        ContextGuard::guard_context(ctx)
    }
}

/// CUDA context handler
#[derive(Debug, PartialEq)]
pub struct Context {
    context_ptr: CUcontext,
}

impl Drop for Context {
    fn drop(&mut self) {
        ffi_call!(cuCtxDestroy_v2, self.context_ptr).expect("Context remove failed");
    }
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// Push to the context stack of this thread
    fn push(&self) {
        ffi_call!(cuCtxPushCurrent_v2, self.context_ptr).expect("Failed to push context");
    }

    /// Pop from the context stack of this thread
    fn pop(&self) {
        let context_ptr = ffi_new!(cuCtxPopCurrent_v2).expect("Failed to pop current context");
        if context_ptr.is_null() {
            panic!("No current context");
        }
        assert!(
            context_ptr == self.context_ptr,
            "Pop must return same pointer"
        );
    }

    /// Create on the top of context stack
    fn create(device: CUdevice) -> Self {
        let context_ptr = ffi_new!(
            cuCtxCreate_v2,
            CUctx_flags_enum::CU_CTX_SCHED_AUTO as u32,
            device
        )
        .expect("Failed to create a new context");
        if context_ptr.is_null() {
            panic!("Cannot crate a new context");
        }
        let ctx = Context { context_ptr };
        ctx.pop();
        ctx
    }

    pub fn version(&self) -> u32 {
        let mut version: u32 = 0;
        ffi_call!(cuCtxGetApiVersion, self.context_ptr, &mut version as *mut _)
            .expect("Failed to get Driver API version");
        version
    }

    /// Block until all tasks to complete.
    pub fn sync(&self) -> Result<()> {
        let _g = self.guard_context();
        ffi_call!(cuCtxSynchronize)?;
        Ok(())
    }
}

impl Contexted for Context {
    fn get_context(&self) -> &Context {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_count() -> Result<()> {
        Device::get_count()?;
        Ok(())
    }

    #[test]
    fn get_zeroth() -> Result<()> {
        Device::nth(0)?;
        Ok(())
    }

    #[test]
    fn create() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        dbg!(&ctx);
        Ok(())
    }
}
