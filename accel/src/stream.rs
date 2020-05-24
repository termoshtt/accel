use crate::{contexted_call, contexted_new, device::*, error::*};
use cuda::*;
use std::{
    future::Future,
    pin::Pin,
    sync::{
        mpsc::{channel, Receiver},
        Arc, Mutex,
    },
    task::{self, Poll, Waker},
};

#[derive(Debug)]
pub struct ThreadFuture {
    recv: Receiver<()>,
    waker: Arc<Mutex<Option<Waker>>>,
}

impl ThreadFuture {
    fn memcpy<'a, T>(
        ctx: &Context,
        from: &'a [T],
        to: &'a mut [T],
    ) -> Pin<Box<dyn Future<Output = ()> + 'a>> {
        assert_eq!(from.len(), to.len());

        let stream = Stream::new(ctx);
        let byte_count = from.len() * std::mem::size_of::<T>();
        unsafe {
            contexted_call!(
                ctx,
                cuMemcpyAsync,
                from.as_ptr() as CUdeviceptr,
                to.as_mut_ptr() as CUdeviceptr,
                byte_count,
                stream.stream
            )
        }
        .unwrap();

        // spawn a thread for waiting memcpy
        let (send, recv) = channel();
        let waker = Arc::new(Mutex::new(None::<Waker>));
        let w = waker.clone();
        std::thread::spawn(move || {
            stream.sync().unwrap();
            send.send(()).unwrap();
            if let Some(waker) = &*w.lock().unwrap() {
                waker.wake_by_ref()
            }
        });
        Box::pin(ThreadFuture { recv, waker })
    }
}

pub fn memcpy_async<'a, T>(
    ctx: &Context,
    from: &'a [T],
    to: &'a mut [T],
) -> Pin<Box<dyn Future<Output = ()> + 'a>> {
    ThreadFuture::memcpy(ctx, from, to)
}

impl Future for ThreadFuture {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut task::Context) -> Poll<Self::Output> {
        *self.waker.lock().unwrap() = Some(cx.waker().clone());
        match self.recv.try_recv() {
            Ok(t) => Poll::Ready(t),
            Err(_) => Poll::Pending,
        }
    }
}

/// Handler for non-blocking CUDA Stream
#[derive(Debug, Contexted)]
pub struct Stream {
    stream: CUstream,
    context: Context,
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
    pub fn new(context: &Context) -> Self {
        let stream = unsafe {
            contexted_new!(
                context,
                cuStreamCreate,
                CUstream_flags::CU_STREAM_NON_BLOCKING as u32
            )
        }
        .expect("Failed to create CUDA stream");
        Stream {
            context: context.clone(),
            stream,
        }
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
    context: Context,
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
    pub fn new(context: &Context) -> Self {
        let event = unsafe {
            contexted_new!(
                context,
                cuEventCreate,
                CUevent_flags_enum::CU_EVENT_BLOCKING_SYNC as u32
            )
        }
        .expect("Failed to create CUDA event");
        Event {
            context: context.clone(),
            event,
        }
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
        let _st = Stream::new(&context);
        Ok(())
    }

    #[test]
    fn trivial_sync() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let mut stream = Stream::new(&context);
        let mut event = Event::new(&context);
        event.record(&mut stream);
        // nothing to be waited
        event.sync()?;
        stream.sync()?;
        Ok(())
    }

    #[tokio::test]
    async fn memcpy_async_host() {
        let device = Device::nth(0).unwrap();
        let ctx = device.create_context();
        let a = vec![1_u32; 12];
        let mut b1 = vec![0_u32; 12];
        let mut b2 = vec![0_u32; 12];
        let mut b3 = vec![0_u32; 12];
        let fut1 = memcpy_async(&ctx, &a, &mut b1);
        let fut2 = memcpy_async(&ctx, &a, &mut b2);
        let fut3 = memcpy_async(&ctx, &a, &mut b3);

        fut3.await;
        fut2.await;
        fut1.await;
    }
}
