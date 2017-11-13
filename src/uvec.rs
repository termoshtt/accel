
use ffi::cuda_runtime as rt;
use super::error::*;

use std::os::raw::*;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::null_mut;

#[derive(Debug)]
pub struct UVec<T> {
    ptr: *mut T,
    n: usize,
}

impl<T> UVec<T> {
    pub fn new(n: usize) -> Result<Self> {
        let mut ptr: *mut c_void = null_mut();
        check(unsafe {
            rt::cudaMallocManaged(&mut ptr as *mut *mut c_void, n, rt::cudaMemAttachGlobal)
        })?;
        Ok(UVec {
            ptr: ptr as *mut T,
            n,
        })
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
        check(unsafe { rt::cudaFree(self.ptr as *mut c_void) }).expect("Free failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uvec_new() {
        let n = 1024;
        let mut uv: UVec<f64> = UVec::new(n).unwrap();
        for i in 0..n {
            uv[i] = 0.0;
        }
    }
}
