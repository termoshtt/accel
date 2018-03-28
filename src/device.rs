use ffi::cudart::*;

use error::*;

pub fn sync() -> Result<()> {
    unsafe { cudaDeviceSynchronize() }.check()
}
