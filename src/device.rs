use ffi::cudart::*;
use error::*;
use std::mem;

pub use ffi::cudart::cudaDeviceProp as DeviceProp;

pub fn sync() -> Result<()> {
    unsafe { cudaDeviceSynchronize() }.check()
}

/// Compute Capability of GPU
///
/// ```
/// let cc40 = ComputeCapability::new(4, 0);
/// let cc35 = ComputeCapability::new(3, 5);
/// let cc30 = ComputeCapability::new(3, 0);
/// assert!(cc30 < cc35);
/// assert!(cc40 > cc35);
/// assert_eq!(cc40, cc40);
/// ```
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct ComputeCapability {
    pub major: i32,
    pub minor: i32,
}

impl ComputeCapability {
    pub fn new(major: i32, minor: i32) -> Self {
        ComputeCapability { major, minor }
    }
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

    pub fn set(id: i32) -> Result<Self> {
        unsafe { cudaSetDevice(id) }.check()?;
        Ok(Device(id))
    }

    pub fn compute_capability(&self) -> Result<ComputeCapability> {
        let prop = self.get_property()?;
        Ok(ComputeCapability::new(prop.major, prop.minor))
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
        unsafe { cudaDeviceGetAttribute(&mut value as *mut _, attr, self.0) }.check()?;
        Ok(value)
    }
}
