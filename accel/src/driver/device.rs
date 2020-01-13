//! Low-level API for device management based on
//! [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE)

use crate::error::*;
use anyhow::Result;
use cuda::*;

#[derive(Debug, PartialEq, PartialOrd)]
pub struct Device(CUdevice);

impl Device {
    pub fn get_count() -> Result<usize> {
        let mut count: i32 = 0;
        unsafe { cuDeviceGetCount(&mut count as *mut i32) }.check()?;
        Ok(count as usize)
    }

    pub fn new(id: i32) -> Result<Self> {
        let mut device: CUdevice = 0;
        unsafe { cuDeviceGet(&mut device as *mut _, id) }.check()?;
        Ok(Device(device))
    }

    pub fn total_memory(&self) -> Result<usize> {
        let mut mem = 0;
        unsafe { cuDeviceTotalMem_v2(&mut mem as *mut _, self.0) }.check()?;
        Ok(mem)
    }

    pub fn get_name(&self) -> Result<String> {
        let mut bytes: Vec<u8> = vec![0_u8; 1024];
        unsafe { cuDeviceGetName(bytes.as_mut_ptr() as *mut i8, 1024, self.0) }.check()?;
        Ok(String::from_utf8(bytes)?)
    }
}
