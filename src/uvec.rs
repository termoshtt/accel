
use ffi::cuda_runtime as rt;
use super::error::Result;

use std::os::raw::*;
use std::ptr::null_mut;

#[derive(Debug)]
pub struct UVec<T> {
    ptr: *mut T,
    n: usize,
}

impl<T> UVec<T> {
    pub fn new(n: usize) -> Result<Self> {
        let mut ptr: *mut c_void = null_mut();
        let flag = rt::cudaMemAttachGlobal;
        cudo!(rt::cudaMallocManaged(&mut ptr as *mut *mut c_void, n, flag));
        Ok(UVec {
            ptr: ptr as *mut T,
            n,
        })
    }
}

impl<T> Drop for UVec<T> {
    fn drop(&mut self) {
        unsafe { rt::cudaFree(self.ptr as *mut c_void) };
    }
}
