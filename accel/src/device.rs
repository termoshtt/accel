use cudart::*;
use error::*;
use std::mem;

pub use cudart::cudaComputeMode as ComputeMode;
pub use cudart::cudaDeviceProp as DeviceProp;

pub fn sync() -> Result<()> {
    unsafe { cudaDeviceSynchronize() }.check()
}

/// Compute Capability of GPU
///
/// ```
/// use accel::device::ComputeCapability;
/// let cc40 = ComputeCapability::new(4, 0);
/// let cc35 = ComputeCapability::new(3, 5);
/// let cc30 = ComputeCapability::new(3, 0);
/// assert!(cc30 < cc35);
/// assert!(cc40 > cc35);
/// assert_eq!(cc40, cc40);
/// ```
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
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

#[derive(PartialEq, Eq, Debug)]
pub struct Device(::std::os::raw::c_int);

impl Device {
    pub fn current() -> Result<Self> {
        let mut id = 0;
        unsafe { cudaGetDevice(&mut id as *mut _) }.check()?;
        Ok(Device(id))
    }

    /// List usbale GPUs
    pub fn usables() -> Result<Vec<Self>> {
        let n = num_devices()? as i32;
        let mut devs = Vec::new();
        for i in 0..n {
            let dev = Device::set(i)?;
            if dev.compute_mode()? != ComputeMode::cudaComputeModeProhibited as i32 {
                devs.push(dev)
            }
        }
        Ok(devs)
    }

    /// Get fastest GPU
    pub fn get_fastest() -> Result<Self> {
        let mut fastest = None;
        let mut max_flops = 0.0;
        for dev in Self::usables()? {
            let flops = dev.flops()?;
            if flops > max_flops {
                max_flops = flops;
                fastest = Some(dev);
            }
        }
        Ok(fastest.expect("No usable GPU"))
    }

    pub fn set(id: i32) -> Result<Self> {
        unsafe { cudaSetDevice(id) }.check()?;
        Ok(Device(id))
    }

    pub fn compute_capability(&self) -> Result<ComputeCapability> {
        let prop = self.get_property()?;
        Ok(ComputeCapability::new(prop.major, prop.minor))
    }

    pub fn name(&self) -> Result<String> {
        let prop = self.get_property()?;
        let name: Vec<u8> = prop
            .name
            .iter()
            .filter_map(|&c| {
                let c = c as u8;
                if c == b'\0' {
                    None
                } else {
                    Some(c)
                }
            })
            .collect();
        Ok(String::from_utf8(name)
            .expect("Invalid GPU name")
            .trim()
            .to_string())
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
        let rate = prop.clockRate as f64 * 1024.0; // clockRate is [kHz]
        Ok(mpc * rate * cores)
    }

    pub fn compute_mode(&self) -> Result<i32> {
        let prop = self.get_property()?;
        Ok(prop.computeMode)
    }

    pub fn get_property(&self) -> Result<DeviceProp> {
        unsafe {
            let mut prop = mem::MaybeUninit::uninit().assume_init();
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
