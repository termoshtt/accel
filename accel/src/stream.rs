use crate::{context::*, error::AccelError, ffi_call_unsafe, ffi_new_unsafe};
use cuda::*;

/// Hanlder for non-blocking CUDA Stream
pub struct Stream<'ctx> {
    _ctx: &'ctx Context,
    stream: CUstream,
}

impl<'ctx> Drop for Stream<'ctx> {
    fn drop(&mut self) {
        ffi_call_unsafe!(cuStreamDestroy_v2, self.stream).expect("Failed to delete CUDA stream");
    }
}

impl<'ctx> Stream<'ctx> {
    /// Create a new non-blocking CUDA stream on the current context
    pub fn new(ctx: &'ctx Context) -> Self {
        ctx.assure_current()
            .expect("Creating a new CUDA stream requires valid and current context");
        let stream = ffi_new_unsafe!(
            cuStreamCreate,
            CUstream_flags::CU_STREAM_NON_BLOCKING as u32
        )
        .expect("Failed to create CUDA stream");
        Stream { _ctx: ctx, stream }
    }

    /// Check all tasks in this stream have been completed
    pub fn query(&self) -> bool {
        match ffi_call_unsafe!(cuStreamQuery, self.stream) {
            Ok(_) => true,
            Err(AccelError::AsyncOperationNotReady) => false,
            Err(e) => panic!("Unknown error is happened while cuStreamQuery: {:?}", e),
        }
    }

    /// Wait until all tasks in this stream have been completed
    pub fn sync(&self) {
        ffi_call_unsafe!(cuStreamSynchronize, self.stream).expect("Failed to sync CUDA stream");
    }

    /// Create a new CUDA event to record all operations in current stream
    pub fn create_event(&self) -> Event {
        todo!()
    }

    /// Wait event to sync another stream
    pub fn wait_event(&self, _event: &Event) {
        todo!()
    }
}

pub struct Event<'stream> {
    stream: &'stream Stream<'stream>,
    event: CUevent,
}

impl<'stream> Drop for Event<'stream> {
    fn drop(&mut self) {
        ffi_call_unsafe!(cuEventDestroy_v2, self.event).expect("Failed to delete CUDA event");
    }
}

impl<'stream> Event<'stream> {
    fn new(_stream: &'stream Stream) -> Self {
        todo!()
    }

    fn record(&self, _stream: &Stream) {
        todo!()
    }

    pub fn query(&self) {
        todo!()
    }

    pub fn sync(&self) {
        todo!()
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
