use ffi::cudart::*;
use error::*;
use std::mem;

pub use ffi::cudart::cudaDeviceProp as DeviceProp;

pub fn sync() -> Result<()> {
    unsafe { cudaDeviceSynchronize() }.check()
}

pub fn num_devices() -> Result<usize> {
    let mut count = 0;
    unsafe { cudaGetDeviceCount(&mut count as *mut _) }.check()?;
    Ok(count as usize)
}

pub struct Device(::std::os::raw::c_int);

impl Device {
    pub fn current() -> Result<Self> {
        let mut id = 0;
        unsafe { cudaGetDevice(&mut id as *mut _) }.check()?;
        Ok(Device(id))
    }

    pub fn get_property(&self) -> Result<DeviceProp> {
        unsafe {
            let mut prop = mem::uninitialized();
            cudaGetDeviceProperties(&mut prop as *mut _, self.0).check()?;
            Ok(prop)
        }
    }

    pub fn get_attr(&self, attr: cudaDeviceAttr) -> Result<i32> {
        let mut value = 0;
        unsafe { cudaDeviceGetAttribute(&mut value as *mut _, attr, self.0) }
            .check()?;
        Ok(value)
    }
}
