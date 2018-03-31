/// Mock internal state of CUDA stream

use core::ptr::null_mut;
use cuda::*;

pub struct Stream(*mut CUstream_st);

impl Stream {
    pub fn blocking() -> Self {
        let mut st = null_mut();
        unsafe { cudaStreamCreateWithFlags(&mut st as *mut _, cudaStreamDefault) }.check();
        Stream(st)
    }
    pub fn non_blocking() -> Self {
        let mut st = null_mut();
        unsafe { cudaStreamCreateWithFlags(&mut st as *mut _, cudaStreamNonBlocking) }.check();
        Stream(st)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { cudaStreamDestroy(self.0) }.check();
    }
}
