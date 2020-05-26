use crate::{contexted_call, contexted_new, device::*, error::*};
use cuda::*;

/// Handler for non-blocking CUDA Stream
#[derive(Debug, Contexted)]
pub struct Stream {
    pub(crate) stream: CUstream,
    context: ContextRef,
}

unsafe impl Sync for Stream {}
unsafe impl Send for Stream {}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Err(e) = unsafe { contexted_call!(self, cuStreamDestroy_v2, self.stream) } {
            log::error!("Failed to delete CUDA stream: {:?}", e);
        }
    }
}

impl Stream {
    /// Create a new non-blocking CUDA stream on the current context
    pub fn new(context: ContextRef) -> Self {
        let stream = unsafe {
            contexted_new!(
                &context,
                cuStreamCreate,
                CUstream_flags::CU_STREAM_NON_BLOCKING as u32
            )
        }
        .expect("Failed to create CUDA stream");
        Stream { context, stream }
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

#[derive(Contexted)]
pub struct Event {
    event: CUevent,
    context: ContextRef,
}

unsafe impl Sync for Event {}
unsafe impl Send for Event {}

impl Drop for Event {
    fn drop(&mut self) {
        if let Err(e) = unsafe { contexted_call!(self, cuEventDestroy_v2, self.event) } {
            log::error!("Failed to delete CUDA event: {:?}", e);
        }
    }
}

impl Event {
    pub fn new(context: ContextRef) -> Self {
        let event = unsafe {
            contexted_new!(
                &context,
                cuEventCreate,
                CUevent_flags_enum::CU_EVENT_BLOCKING_SYNC as u32
            )
        }
        .expect("Failed to create CUDA event");
        Event { context, event }
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
        let context = device.create_context();
        let _st = Stream::new(context.get_ref());
        Ok(())
    }

    #[test]
    fn trivial_sync() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let mut stream = Stream::new(context.get_ref());
        let mut event = Event::new(context.get_ref());
        event.record(&mut stream);
        // nothing to be waited
        event.sync()?;
        stream.sync()?;
        Ok(())
    }
}
