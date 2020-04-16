use crate::{contexted_call, contexted_new, device::*, error::*};
use cuda::*;

/// Handler for non-blocking CUDA Stream
pub struct Stream<'ctx> {
    ctx: &'ctx Context,
    pub(crate) stream: CUstream,
}

impl<'ctx> Drop for Stream<'ctx> {
    fn drop(&mut self) {
        if let Err(e) = unsafe { contexted_call!(self, cuStreamDestroy_v2, self.stream) } {
            log::error!("Failed to delete CUDA stream: {:?}", e);
        }
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
        let stream = unsafe {
            contexted_new!(
                ctx,
                cuStreamCreate,
                CUstream_flags::CU_STREAM_NON_BLOCKING as u32
            )
        }
        .expect("Failed to create CUDA stream");
        Stream { ctx, stream }
    }

    /// Check all tasks in this stream have been completed
    pub fn query(&self) -> bool {
        match unsafe { contexted_call!(self, cuStreamQuery, self.stream) } {
            Ok(_) => true,
            Err(AccelError::AsyncOperationNotReady) => false,
            Err(e) => panic!("Unknown error is happened while cuStreamQuery: {:?}", e),
        }
    }

    /// Wait until all tasks in this stream have been completed
    pub fn sync(&self) -> Result<()> {
        unsafe { contexted_call!(self, cuStreamSynchronize, self.stream) }?;
        Ok(())
    }

    /// Wait event to sync another stream
    pub fn wait_event(&mut self, event: &Event) {
        unsafe { contexted_call!(self, cuStreamWaitEvent, self.stream, event.event, 0) }
            .expect("Failed to register an CUDA event waiting on CUDA stream");
    }
}

pub struct Event<'ctx> {
    ctx: &'ctx Context,
    event: CUevent,
}

impl<'ctx> Drop for Event<'ctx> {
    fn drop(&mut self) {
        if let Err(e) = unsafe { contexted_call!(self, cuEventDestroy_v2, self.event) } {
            log::error!("Failed to delete CUDA event: {:?}", e);
        }
    }
}

impl<'ctx> Contexted for Event<'ctx> {
    fn get_context(&self) -> &Context {
        &self.ctx
    }
}

impl<'ctx> Event<'ctx> {
    pub fn new(ctx: &'ctx Context) -> Self {
        let event = unsafe {
            contexted_new!(
                ctx,
                cuEventCreate,
                CUevent_flags_enum::CU_EVENT_BLOCKING_SYNC as u32
            )
        }
        .expect("Failed to create CUDA event");
        Event { ctx, event }
    }

    pub fn record(&mut self, stream: &mut Stream) {
        unsafe { contexted_call!(self, cuEventRecord, self.event, stream.stream) }
            .expect("Failed to set event record");
    }

    /// Query if the event has occur, returns true if already occurs
    pub fn query(&self) -> bool {
        match unsafe { contexted_call!(self, cuEventQuery, self.event) } {
            Ok(_) => true,
            Err(AccelError::AsyncOperationNotReady) => false,
            Err(e) => panic!("Unknown error occurs while cuEventQuery: {:?}", e),
        }
    }

    /// Wait until the event occurs with blocking
    pub fn sync(&self) -> Result<()> {
        unsafe { contexted_call!(self, cuEventSynchronize, self.event) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _st = Stream::new(&ctx);
        Ok(())
    }

    #[test]
    fn trivial_sync() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let mut stream = Stream::new(&ctx);
        let mut event = Event::new(&ctx);
        event.record(&mut stream);
        // nothing to be waited
        event.sync()?;
        stream.sync()?;
        Ok(())
    }
}
