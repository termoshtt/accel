use super::*;
use crate::*;
use cuda::*;
use std::{
    ffi::c_void,
    ops::{Deref, DerefMut},
    sync::Arc,
};

pub struct RegisteredMemory<'a, T> {
    ctx: Arc<Context>,
    mem: &'a mut [T],
}

impl<T> Deref for RegisteredMemory<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.mem
    }
}

impl<T> DerefMut for RegisteredMemory<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.mem
    }
}

impl<T> Drop for RegisteredMemory<'_, T> {
    fn drop(&mut self) {
        if let Err(e) = unsafe {
            contexted_call!(
                &self.ctx,
                cuMemHostUnregister,
                self.mem.as_mut_ptr() as *mut c_void
            )
        } {
            log::error!("Failed to unregister memory: {:?}", e);
        }
    }
}

impl<'a, T: Scalar> RegisteredMemory<'a, T> {
    pub fn new(ctx: Arc<Context>, mem: &'a mut [T]) -> Self {
        unsafe {
            contexted_call!(
                &ctx,
                cuMemHostRegister_v2,
                mem.as_mut_ptr() as *mut c_void,
                mem.len() * T::size_of(),
                0
            )
        }
        .expect("Failed to register host memory into CUDA memory system");
        Self { ctx, mem }
    }
}

impl<T: Scalar> Memory for RegisteredMemory<'_, T> {
    type Elem = T;

    fn head_addr(&self) -> *const T {
        self.mem.as_ptr()
    }

    fn head_addr_mut(&mut self) -> *mut T {
        self.mem.as_mut_ptr()
    }

    fn num_elem(&self) -> usize {
        self.mem.len()
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::Registered
    }
}

impl<T> Contexted for RegisteredMemory<'_, T> {
    fn get_context(&self) -> Arc<Context> {
        self.ctx.clone()
    }
}
