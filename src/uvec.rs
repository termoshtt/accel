
use ffi::cuda_runtime as rt;
use super::error::*;

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
        check(unsafe {
            rt::cudaMallocManaged(&mut ptr as *mut *mut c_void, n, rt::cudaMemAttachGlobal)
        })?;
        Ok(UVec {
            ptr: ptr as *mut T,
            n,
        })
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
        let _uv: UVec<f64> = UVec::new(1024).unwrap();
    }
}
