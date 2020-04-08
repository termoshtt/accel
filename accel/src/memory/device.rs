//! Device and Host memory handlers

use super::*;
use crate::{device::*, ffi_call, ffi_new};
use cuda::*;
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use cuda::CUmemAttach_flags_enum as AttachFlag;

/// Memory allocated on the device.
pub struct DeviceMemory<'ctx, T> {
    ptr: CUdeviceptr,
    size: usize,
    context: &'ctx Context,
    phantom: PhantomData<T>,
}

impl<'ctx, T> Drop for DeviceMemory<'ctx, T> {
    fn drop(&mut self) {
        if let Err(e) = ffi_call!(cuMemFree_v2, self.ptr) {
            log::error!("Failed to free device memory: {:?}", e);
        }
    }
}

impl<'ctx, T> Deref for DeviceMemory<'ctx, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'ctx, T> DerefMut for DeviceMemory<'ctx, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'ctx, T> Memory for DeviceMemory<'ctx, T> {
    type Elem = T;
    fn head_addr(&self) -> *const T {
        self.ptr as _
    }
    fn byte_size(&self) -> usize {
        self.size * std::mem::size_of::<T>()
    }
}

impl<'ctx, T> MemoryMut for DeviceMemory<'ctx, T> {
    fn head_addr_mut(&mut self) -> *mut T {
        self.ptr as _
    }
}

impl<'ctx, T> Continuous for DeviceMemory<'ctx, T> {
    fn length(&self) -> usize {
        self.size
    }
    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.head_addr(), self.size) }
    }
}

impl<'ctx, T> ContinuousMut for DeviceMemory<'ctx, T> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.head_addr_mut(), self.size) }
    }
}

impl<'ctx, T> Managed for DeviceMemory<'ctx, T> {}

impl<'ctx, T> Contexted for DeviceMemory<'ctx, T> {
    fn get_context(&self) -> &Context {
        &self.context
    }
}

impl<'ctx, T> DeviceMemory<'ctx, T> {
    /// Allocate a new device memory with `size` byte length by [cuMemAllocManaged].
    /// This memory is managed by the unified memory system.
    ///
    /// Panic
    /// ------
    /// - when given context is not current
    /// - allocation failed including `size == 0` case
    ///
    /// [cuMemAllocManaged]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467
    ///
    pub fn new(context: &'ctx Context, size: usize) -> Self {
        assert!(size > 0, "Zero-sized malloc is forbidden");
        let _gurad = context.guard_context();
        let ptr = ffi_new!(
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
        let mut mem = DeviceMemory::<i32>::new(&ctx, 12);
        assert_eq!(mem.len(), 12);
        assert_eq!(mem.byte_size(), 12 * 4 /* size of i32 */);
        let sl = mem.as_mut_slice();
        sl[0] = 3;
        Ok(())
    }

    #[should_panic(expected = "Zero-sized malloc is forbidden")]
    #[test]
    fn device_new_zero() {
        let device = Device::nth(0).unwrap();
        let ctx = device.create_context();
        let _a = DeviceMemory::<i32>::new(&ctx, 0);
    }
}
