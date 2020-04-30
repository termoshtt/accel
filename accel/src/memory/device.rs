//! Device and Host memory handlers

use super::*;
use crate::*;
use cuda::*;
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use cuda::CUmemAttach_flags_enum as AttachFlag;

/// Memory allocated on the device.
pub struct DeviceMemory<T> {
    ptr: CUdeviceptr,
    size: usize,
    context: Arc<Context>,
    phantom: PhantomData<T>,
}

impl<T> Drop for DeviceMemory<T> {
    fn drop(&mut self) {
        if let Err(e) = unsafe { contexted_call!(self, cuMemFree_v2, self.ptr) } {
            log::error!("Failed to free device memory: {:?}", e);
        }
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
}

impl<T: Scalar> Memcpy<Self> for DeviceMemory<T> {
    fn copy_from(&mut self, src: &Self) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        unsafe {
            contexted_call!(
                &self.get_context(),
                cuMemcpyDtoD_v2,
                self.as_mut_ptr() as CUdeviceptr,
                src.as_ptr() as CUdeviceptr,
                self.num_elem() * T::size_of()
            )
        }
        .expect("memcpy between Device memories failed")
    }
}

impl<T: Scalar> Memcpy<PageLockedMemory<T>> for DeviceMemory<T> {
    fn copy_from(&mut self, src: &PageLockedMemory<T>) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        unsafe {
            contexted_call!(
                &self.get_context(),
                cuMemcpyHtoD_v2,
                self.as_mut_ptr() as CUdeviceptr,
                src.as_ptr() as *mut _,
                self.num_elem() * T::size_of()
            )
        }
        .expect("memcpy from Page-locked host memory to Device memory failed")
    }
}

impl<T: Scalar> Memcpy<RegisteredMemory<'_, T>> for DeviceMemory<T> {
    fn copy_from(&mut self, src: &RegisteredMemory<'_, T>) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        unsafe {
            contexted_call!(
                &self.get_context(),
                cuMemcpyHtoD_v2,
                self.as_mut_ptr() as CUdeviceptr,
                src.as_ptr() as *mut _,
                self.num_elem() * T::size_of()
            )
        }
        .expect("memcpy from registered host memory to Device memory failed")
    }
}

impl<T: Scalar> Memset for DeviceMemory<T> {
    fn set(&mut self, value: T) {
        match T::size_of() {
            1 => unsafe {
                contexted_call!(
                    &self.get_context(),
                    cuMemsetD8_v2,
                    self.head_addr_mut() as CUdeviceptr,
                    value.to_le_u8().unwrap(),
                    self.num_elem()
                )
            }
            .expect("memset failed for 8-bit scalar"),
            2 => unsafe {
                contexted_call!(
                    &self.get_context(),
                    cuMemsetD16_v2,
                    self.head_addr_mut() as CUdeviceptr,
                    value.to_le_u16().unwrap(),
                    self.num_elem()
                )
            }
            .expect("memset failed for 16-bit scalar"),
            4 => unsafe {
                contexted_call!(
                    &self.get_context(),
                    cuMemsetD32_v2,
                    self.head_addr_mut() as CUdeviceptr,
                    value.to_le_u32().unwrap(),
                    self.num_elem()
                )
            }
            .expect("memset failed for 64-bit scalar"),
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

impl<T: Scalar> Managed for DeviceMemory<T> {}

impl<T> Contexted for DeviceMemory<T> {
    fn get_context(&self) -> Arc<Context> {
        self.context.clone()
    }
}

impl<T: Scalar> Allocatable for DeviceMemory<T> {
    type Shape = usize;
    unsafe fn uninitialized(context: Arc<Context>, size: usize) -> Self {
        assert!(size > 0, "Zero-sized malloc is forbidden");
        let ptr = contexted_new!(
            &context,
            cuMemAllocManaged,
            size * std::mem::size_of::<T>(),
            AttachFlag::CU_MEM_ATTACH_GLOBAL as u32
        )
        .expect("Cannot allocate device memory");
        DeviceMemory {
            ptr,
            size,
            context,
            phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::*;

    #[test]
    fn device() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let mut mem = DeviceMemory::<i32>::zeros(ctx, 12);
        assert_eq!(mem.num_elem(), 12);
        let sl = mem.as_mut_slice();
        sl[0] = 3;
        Ok(())
    }

    #[should_panic(expected = "Zero-sized malloc is forbidden")]
    #[test]
    fn device_new_zero() {
        let device = Device::nth(0).unwrap();
        let ctx = device.create_context();
        let _a = DeviceMemory::<i32>::zeros(ctx, 0);
    }
}
