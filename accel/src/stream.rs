use crate::{contexted_call, contexted_new, device::*, error::*};
use cuda::*;
use lazy_static::lazy_static;
use std::{
    collections::{hash_map::Entry, HashMap},
    ffi::c_void,
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    task::{self, Poll, Waker},
};

lazy_static! {
    static ref WAKER: Arc<Mutex<HashMap<usize, Waker>>> = Arc::new(Mutex::new(HashMap::new()));
}
static mut WAKER_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn wake_nth(arg: *mut c_void) {
    let id = (arg as *const usize).as_ref().unwrap();
    if let Ok(mut map) = WAKER.lock() {
        if let Some(waker) = map.remove(id) {
            waker.wake()
        }
    } else {
        log::error!("Error while locking mutex, maybe another thread panics with lock");
    }
}

#[derive(Debug)]
pub struct MemcpyFuture<'a, T> {
    id: usize,
    from: &'a [T],
    to: &'a mut [T],
    stream: Stream,
}

impl<'a, T> MemcpyFuture<'a, T> {
    fn new(ctx: &Context, from: &'a [T], to: &'a mut [T]) -> Result<Pin<Box<Self>>> {
        assert_eq!(from.len(), to.len());
        let id = unsafe { WAKER_ID_COUNTER.fetch_add(1, Ordering::Relaxed) };
        let byte_count = from.len() * std::mem::size_of::<T>();
        let stream = Stream::new(ctx);
        unsafe {
            contexted_call!(
                ctx,
                cuMemcpyAsync,
                from.as_ptr() as CUdeviceptr,
                to.as_mut_ptr() as CUdeviceptr,
                byte_count,
                stream.stream
            )?;
            contexted_call!(
                ctx,
                cuLaunchHostFunc,
                stream.stream,
                Some(wake_nth),
                &id as *const usize as *mut c_void
            )?;
        }

        Ok(Box::pin(MemcpyFuture {
            id,
            from,
            to,
            stream,
        }))
    }
}

pub fn memcpy_async<'a, T>(
    ctx: &Context,
    from: &'a [T],
    to: &'a mut [T],
) -> Pin<Box<MemcpyFuture<'a, T>>> {
    MemcpyFuture::new(ctx, from, to).unwrap()
}

impl<'a, T> Future for MemcpyFuture<'a, T> {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut task::Context) -> Poll<Self::Output> {
        let mut map = WAKER
            .lock()
            .expect("Cannot lock global mutex, another thread paniced with lock");
        match map.entry(self.id) {
            Entry::Vacant(v) => {
                if self.stream.query() {
                    Poll::Ready(())
                } else {
                    v.insert(cx.waker().clone());
                    Poll::Pending
                }
            }
            Entry::Occupied(mut o) => {
                o.insert(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

/// Handler for non-blocking CUDA Stream
#[derive(Debug, Contexted)]
pub struct Stream {
    stream: CUstream,
    context: Context,
}

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
        let mut b = vec![0_u32; 12];
        memcpy_async(&ctx, &a, &mut b).await;
    }
}
