use super::{error::*, ffi_call, ffi_call_unsafe};
use cudart::*;

use std::mem::size_of;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::os::raw::*;
use std::ptr::null_mut;
use std::slice::{from_raw_parts, from_raw_parts_mut};

/// An implementation of Vec based on Uniform Memory
#[derive(Debug)]
pub struct UVec<T> {
    ptr: *mut T,
    n: usize,
}

impl<T> UVec<T> {
    pub unsafe fn uninitialized(n: usize) -> Result<Self> {
        let mut ptr: *mut c_void = null_mut();
        ffi_call!(
            cudaMallocManaged,
            &mut ptr as *mut *mut c_void,
            n * size_of::<T>(),
            cudaMemAttachGlobal
        )?;
        Ok(UVec {
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

    /// Recast to Rust's immutable slice
    pub fn as_slice(&self) -> &[T] {
        unsafe { from_raw_parts(self.ptr, self.n) }
    }

    /// Recast to Rust's mutable slice
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { from_raw_parts_mut(self.ptr, self.n) }
    }
}

impl<T> Deref for UVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for UVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}

impl<T> Index<usize> for UVec<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.ptr.offset(index as isize) }
    }
}

impl<T> IndexMut<usize> for UVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.ptr.offset(index as isize) }
    }
}

impl<T> Drop for UVec<T> {
    fn drop(&mut self) {
        ffi_call_unsafe!(cudaFree, self.ptr as *mut c_void).expect("Free failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uvec_new() {
        // zero-filled on GPU
        let v: UVec<u64> = UVec::new(128).unwrap();
        assert!(v.iter().all(|v| *v == 0));
    }
}
