use super::*;
use crate::{error::Result, *};
use cuda::*;
use std::{
    ffi::c_void,
    ops::{Deref, DerefMut},
};

#[derive(Contexted, Debug)]
pub struct RegisteredMemory<'a, T> {
    context: Context,
    data: &'a mut [T],
}

unsafe impl<T> Sync for RegisteredMemory<'_, T> {}
unsafe impl<T> Send for RegisteredMemory<'_, T> {}

impl<T> Deref for RegisteredMemory<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.data
    }
}

impl<T> DerefMut for RegisteredMemory<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.data
    }
}

impl<T: Scalar> PartialEq for RegisteredMemory<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<T: Scalar> PartialEq<[T]> for RegisteredMemory<'_, T> {
    fn eq(&self, other: &[T]) -> bool {
        self.as_slice().eq(other)
    }
}

impl<T> Drop for RegisteredMemory<'_, T> {
    fn drop(&mut self) {
        if let Err(e) = unsafe {
            contexted_call!(
                &self.context,
                cuMemHostUnregister,
                self.data.as_mut_ptr() as *mut c_void
            )
        } {
            log::error!("Failed to unregister memory: {:?}", e);
        }
    }
}

impl<'a, T: Scalar> RegisteredMemory<'a, T> {
    pub fn new(context: &Context, data: &'a mut [T]) -> Self {
        unsafe {
            contexted_call!(
                context,
                cuMemHostRegister_v2,
                data.as_mut_ptr() as *mut c_void,
                data.len() * T::size_of(),
                0
            )
        }
        .expect("Failed to register host memory into CUDA memory system");
        Self {
            context: context.clone(),
            data,
        }
    }
}

impl<T: Scalar> Memory for RegisteredMemory<'_, T> {
    type Elem = T;

    fn head_addr(&self) -> *const T {
        self.data.as_ptr()
    }

    fn head_addr_mut(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    fn num_elem(&self) -> usize {
        self.data.len()
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::Host
    }
}

impl<T: Scalar> Memset for RegisteredMemory<'_, T> {
    fn set(&mut self, value: Self::Elem) {
        self.iter_mut().for_each(|v| *v = value);
    }
}

impl<T: Scalar> Continuous for RegisteredMemory<'_, T> {
    fn as_slice(&self) -> &[T] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

impl<'arg, 'a: 'arg, T: Scalar> DeviceSend for &'arg RegisteredMemory<'a, T> {
    type Target = *const T;
    fn as_kernel_parameter(&self) -> *mut c_void {
        self.data.as_kernel_parameter()
    }
}

impl<'arg, 'a: 'arg, T: Scalar> DeviceSend for &'arg mut RegisteredMemory<'a, T> {
    type Target = *mut T;
    fn as_kernel_parameter(&self) -> *mut c_void {
        self.data.as_kernel_parameter()
    }
}
