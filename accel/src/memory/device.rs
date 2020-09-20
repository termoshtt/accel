//! Device and Host memory handlers

use super::*;
use crate::{error::*, *};
use cuda::*;
use std::{
    fmt,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use cuda::CUmemAttach_flags_enum as AttachFlag;

/// Memory allocated on the device.
#[derive(Contexted)]
pub struct DeviceMemory<T> {
    ptr: CUdeviceptr,
    size: usize,
    context: Context,
    phantom: PhantomData<T>,
}

unsafe impl<T> Sync for DeviceMemory<T> {}
unsafe impl<T> Send for DeviceMemory<T> {}

impl<T> Drop for DeviceMemory<T> {
    fn drop(&mut self) {
        if let Err(e) = unsafe { contexted_call!(self, cuMemFree_v2, self.ptr) } {
            log::error!("Failed to free device memory: {:?}", e);
        }
    }
}

impl<T: Scalar> fmt::Debug for DeviceMemory<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceMemory")
            .field("context", &self.context)
            .field("data", &self.as_slice())
            .finish()
    }
}

impl<T> Deref for DeviceMemory<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr as _, self.size) }
    }
}

impl<T> DerefMut for DeviceMemory<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as _, self.size) }
    }
}

impl<T: Scalar> PartialEq for DeviceMemory<T> {
    fn eq(&self, other: &Self) -> bool {
        // FIXME should be tested on device
        self.as_slice().eq(other.as_slice())
    }
}

impl<T: Scalar> PartialEq<[T]> for DeviceMemory<T> {
    fn eq(&self, other: &[T]) -> bool {
        // FIXME should be tested on device
        self.as_slice().eq(other)
    }
}

impl<T: Scalar> Memory for DeviceMemory<T> {
    type Elem = T;
    fn head_addr(&self) -> *const T {
        self.ptr as _
    }

    fn head_addr_mut(&mut self) -> *mut T {
        self.ptr as _
    }

    fn num_elem(&self) -> usize {
        self.size
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::Device
    }

    fn set(&mut self, value: T) {
        match T::size_of() {
            1 => unsafe {
                contexted_call!(
                    self,
                    cuMemsetD8_v2,
                    self.head_addr_mut() as CUdeviceptr,
                    value.to_le_u8().unwrap(),
                    self.num_elem()
                )
            }
            .expect("memset failed for 8-bit scalar"),
            2 => unsafe {
                contexted_call!(
                    self,
                    cuMemsetD16_v2,
                    self.head_addr_mut() as CUdeviceptr,
                    value.to_le_u16().unwrap(),
                    self.num_elem()
                )
            }
            .expect("memset failed for 16-bit scalar"),
            4 => unsafe {
                contexted_call!(
                    self,
                    cuMemsetD32_v2,
                    self.head_addr_mut() as CUdeviceptr,
                    value.to_le_u32().unwrap(),
                    self.num_elem()
                )
            }
            .expect("memset failed for 32-bit scalar"),
            _ => {
                unimplemented!("memset for Device memory is only supported for 8/16/32-bit scalars")
            }
        }
    }
}

impl<T: Scalar> Continuous for DeviceMemory<T> {
    fn as_slice(&self) -> &[T] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

impl<T: Scalar> Allocatable for DeviceMemory<T> {
    type Shape = usize;
    unsafe fn uninitialized(context: &Context, size: usize) -> Self {
        assert!(size > 0, "Zero-sized malloc is forbidden");
        let ptr = contexted_new!(
            context,
            cuMemAllocManaged,
            size * std::mem::size_of::<T>(),
            AttachFlag::CU_MEM_ATTACH_GLOBAL as u32
        )
        .expect("Cannot allocate device memory");
        DeviceMemory {
            ptr,
            size,
            context: context.clone(),
            phantom: PhantomData,
        }
    }
}

impl<'arg, T: Scalar> DeviceSend for &'arg DeviceMemory<T> {
    type Target = *const T;
    fn as_kernel_parameter(&self) -> *mut c_void {
        &self.ptr as *const CUdeviceptr as *mut c_void
    }
}

impl<'arg, T: Scalar> DeviceSend for &'arg mut DeviceMemory<T> {
    type Target = *mut T;
    fn as_kernel_parameter(&self) -> *mut c_void {
        &self.ptr as *const CUdeviceptr as *mut c_void
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_mut_slice() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let mut mem = DeviceMemory::<i32>::zeros(&context, 12);
        let sl = mem.as_mut_slice();
        sl[0] = 3; // test if accessible from host
        assert_eq!(sl.num_elem(), 12);
        Ok(())
    }

    #[should_panic(expected = "Zero-sized malloc is forbidden")]
    #[test]
    fn device_new_zero() {
        let device = Device::nth(0).unwrap();
        let context = device.create_context();
        let _a = DeviceMemory::<i32>::zeros(&context, 0);
    }
}
