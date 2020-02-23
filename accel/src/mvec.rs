use super::{error::*, ffi_call, ffi_call_unsafe};
use anyhow::Result;
use cudart::*;
use std::mem::size_of;
use std::os::raw::*;
use std::ptr::null_mut;

/// An implementation of Vec based on Managed Memory
#[derive(Debug)]
pub struct MVec<T> {
    ptr: *mut T,
    n: usize,
}

impl<T: Copy> MVec<T> {
    pub unsafe fn uninitialized(n: usize) -> Result<Self> {
        let mut ptr: *mut c_void = null_mut();
        ffi_call!(cudaMalloc, &mut ptr as *mut *mut c_void, n * size_of::<T>())?;
        Ok(MVec {
            ptr: ptr as *mut T,
            n,
        })
    }

    pub fn fill_zero(&mut self) -> Result<()> {
        ffi_call_unsafe!(
            cudaMemset,
            self.ptr as *mut c_void,
            0,
            self.n * size_of::<T>()
        )?;
        Ok(())
    }

    pub fn new(n: usize) -> Result<Self> {
        let mut v = unsafe { Self::uninitialized(n) }?;
        v.fill_zero()?;
        Ok(v)
    }

    /// Returns a raw pointer to the buffer
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Returns a raw mutable pointer to the buffer
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Load data from host_vector into device_vector
    pub fn set(&mut self, data: &[T]) -> Result<()> {
        assert!(self.len() == data.len());
        ffi_call_unsafe!(
            cudaMemcpy,
            self.ptr as *mut c_void,
            data.as_ptr() as *const c_void,
            self.n * size_of::<T>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice
        )?;
        Ok(())
    }

    /// Load data from device_vector into host_vector
    pub fn get(&self, buffer: &mut [T]) -> Result<()> {
        assert!(self.len() == buffer.len());
        ffi_call_unsafe!(
            cudaMemcpy,
            buffer.as_mut_ptr() as *mut c_void,
            self.ptr as *const c_void,
            self.n * size_of::<T>(),
            cudaMemcpyKind::cudaMemcpyDeviceToHost
        )?;
        Ok(())
    }

    /// Returns the length of the buffer
    pub fn len(&self) -> usize {
        self.n
    }
}

impl<T> Drop for MVec<T> {
    fn drop(&mut self) {
        ffi_call_unsafe!(cudaFree, self.ptr as *mut c_void).expect("Free failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mvec_new() {
        // zero-filled on GPU
        let v: MVec<u64> = MVec::new(128).unwrap();
        let mut buff = vec![1; 128]; // create buffer filled with 1
        v.get(buff.as_mut_slice()).unwrap();
        assert!(buff.iter().all(|v| *v == 0));
    }
}
