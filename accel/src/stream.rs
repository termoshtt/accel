use crate::{context::*, error::AccelError, ffi_call_unsafe, ffi_new_unsafe};
use cuda::*;

pub struct Stream {
    stream: CUstream,
}

impl Drop for Stream {
    fn drop(&mut self) {
        ffi_call_unsafe!(cuStreamDestroy_v2, self.stream).expect("Failed to delete CUDA stream");
    }
}

impl Stream {
    fn init(ctx: &Context, flag: CUstream_flags) -> Self {
        ctx.assure_current()
            .expect("Creating a new CUDA stream requires valid and current context");
        let stream =
            ffi_new_unsafe!(cuStreamCreate, flag as u32).expect("Failed to create CUDA stream");
        Stream { stream }
    }

    pub fn new(ctx: &Context) -> Self {
        Self::init(ctx, CUstream_flags::CU_STREAM_DEFAULT)
    }

    /// Stream does not synchronize with stream 0 (the NULL stream)
    pub fn non_blocking(ctx: &Context) -> Self {
        Self::init(ctx, CUstream_flags::CU_STREAM_NON_BLOCKING)
    }

    pub fn is_completed(&self) -> bool {
        match ffi_call_unsafe!(cuStreamQuery, self.stream) {
            Ok(_) => true,
            Err(AccelError::AsyncOperationNotReady) => false,
            Err(e) => panic!("Unknown error is happened while cuStreamQuery: {:?}", e),
        }
    }

    pub fn sync(&self) {
        ffi_call_unsafe!(cuStreamSynchronize, self.stream).expect("Failed to sync CUDA stream");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{device::*, error::*};

    #[test]
    fn new() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let _st = Stream::new(&ctx);
        Ok(())
    }

    #[test]
    fn non_blocking() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let _st = Stream::non_blocking(&ctx);
        Ok(())
    }
}
