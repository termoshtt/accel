use crate::{device::*, error::AccelError, ffi_call, ffi_new};
use cuda::*;

/// Hanlder for non-blocking CUDA Stream
pub struct Stream<'ctx> {
    ctx: &'ctx Context,
    stream: CUstream,
}

impl<'ctx> Drop for Stream<'ctx> {
    fn drop(&mut self) {
        ffi_call!(cuStreamDestroy_v2, self.stream).expect("Failed to delete CUDA stream");
    }
}

impl<'ctx> Contexted for Stream<'ctx> {
    fn get_context(&self) -> &Context {
        &self.ctx
    }
}

impl<'ctx> Stream<'ctx> {
    /// Create a new non-blocking CUDA stream on the current context
    pub fn new(ctx: &'ctx Context) -> Self {
        let _gurad = ctx.guard_context();
        let stream = ffi_new!(
            cuStreamCreate,
            CUstream_flags::CU_STREAM_NON_BLOCKING as u32
        )
        .expect("Failed to create CUDA stream");
        Stream { ctx, stream }
    }

    /// Check all tasks in this stream have been completed
    pub fn query(&self) -> bool {
        let _g = self.guard_context();
        match ffi_call!(cuStreamQuery, self.stream) {
            Ok(_) => true,
            Err(AccelError::AsyncOperationNotReady) => false,
            Err(e) => panic!("Unknown error is happened while cuStreamQuery: {:?}", e),
        }
    }

    /// Wait until all tasks in this stream have been completed
    pub fn sync(&self) {
        let _g = self.guard_context();
        ffi_call!(cuStreamSynchronize, self.stream).expect("Failed to sync CUDA stream");
    }

    /// Create a new CUDA event to record all operations in current stream
    pub fn create_event(&mut self) -> Event {
        let e = Event::new(self.get_context());
        e.record(self);
        e
    }

    /// Wait event to sync another stream
    pub fn wait_event(&mut self, event: &Event) {
        let _g = self.guard_context();
        ffi_call!(cuStreamWaitEvent, self.stream, event.event, 0)
            .expect("Failed to register an CUDA event waiting on CUDA stream");
    }
}

pub struct Event<'ctx> {
    ctx: &'ctx Context,
    event: CUevent,
}

impl<'ctx> Drop for Event<'ctx> {
    fn drop(&mut self) {
        ffi_call!(cuEventDestroy_v2, self.event).expect("Failed to delete CUDA event");
    }
}

impl<'ctx> Contexted for Event<'ctx> {
    fn get_context(&self) -> &Context {
        &self.ctx
    }
}

impl<'ctx> Event<'ctx> {
    fn new(ctx: &'ctx Context) -> Self {
        let _gurad = ctx.guard_context();
        let event = ffi_new!(
            cuEventCreate,
            CUevent_flags_enum::CU_EVENT_BLOCKING_SYNC as u32
        )
        .expect("Failed to create CUDA event");
        Event { ctx, event }
    }

    /// Do not capture the lifetime of stream.
    /// i.e. The stream may be dropped while event lives.
    fn record(&self, stream: &Stream) {
        let _g = self.guard_context();
        ffi_call!(cuEventRecord, self.event, stream.stream).expect("Failed to set event record");
    }

    /// Query if the event has occured, returns true if already occurs
    pub fn query(&self) -> bool {
        let _g = self.guard_context();
        match ffi_call!(cuEventQuery, self.event) {
            Ok(_) => true,
            Err(AccelError::AsyncOperationNotReady) => false,
            Err(e) => panic!("Unknown error occurs while cuEventQuery: {:?}", e),
        }
    }

    /// Wait until the event occurs with blocking
    pub fn sync(&self) {
        let _g = self.guard_context();
        ffi_call!(cuEventQuery, self.event).expect("Failed to sync CUDA event");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::*;

    #[test]
    fn new() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _st = Stream::new(&ctx);
        Ok(())
    }
}
