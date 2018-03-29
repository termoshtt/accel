use ffi::cudart::*;
use error::*;
use std::mem;

pub use ffi::cudart::cudaDeviceProp as DeviceProp;
pub use ffi::cudart::cudaComputeMode as ComputeMode;

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

    pub fn cores(&self) -> Result<u32> {
        let cc = self.compute_capability()?;
        Ok(match (cc.major, cc.minor) {
            (3, 0) => 192, // Kepler Generation (SM 3.0) GK10x class
            (3, 2) => 192, // Kepler Generation (SM 3.2) GK10x class
            (3, 5) => 192, // Kepler Generation (SM 3.5) GK11x class
            (3, 7) => 192, // Kepler Generation (SM 3.7) GK21x class
            (5, 0) => 128, // Maxwell Generation (SM 5.0) GM10x class
            (5, 2) => 128, // Maxwell Generation (SM 5.2) GM20x class
            (5, 3) => 128, // Maxwell Generation (SM 5.3) GM20x class
            (6, 0) => 64,  // Pascal Generation (SM 6.0) GP100 class
            (6, 1) => 128, // Pascal Generation (SM 6.1) GP10x class
            (6, 2) => 128, // Pascal Generation (SM 6.2) GP10x class
            (7, 0) => 64,  // Volta Generation (SM 7.0) GV100 class
            _ => unreachable!("Unsupported Core"),
        })
    }

    pub fn flops(&self) -> Result<f64> {
        let prop = self.get_property()?;
        let cores = self.cores()? as f64;
        let mpc = prop.multiProcessorCount as f64;
        let rate = prop.clockRate as f64;
        Ok(mpc * rate * cores)
    }

    pub fn compute_mode(&self) -> Result<ComputeMode> {
        let prop = self.get_property()?;
        Ok(prop.computeMode)
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
