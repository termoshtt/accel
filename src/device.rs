
use ffi::cuda_runtime::*;

use error::*;

pub fn sync() -> Result<()> {
    unsafe { cudaDeviceSynchronize() }.check()
}
