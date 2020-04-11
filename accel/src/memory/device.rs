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
        unsafe { std::slice::from_raw_parts(self.ptr as _, self.size) }
    }
}

impl<'ctx, T> DerefMut for DeviceMemory<'ctx, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as _, self.size) }
    }
}

impl<'ctx, T: Copy> Memory for DeviceMemory<'ctx, T> {
    type Elem = T;
    fn head_addr(&self) -> *const T {
        self.ptr as _
    }
    fn byte_size(&self) -> usize {
        self.size * std::mem::size_of::<T>()
    }
    fn try_as_slice(&self) -> Option<&[T]> {
        Some(self.as_slice())
    }
    fn try_get_context(&self) -> Option<&Context> {
        Some(self.get_context())
    }
    fn memory_type(&self) -> MemoryType {
        MemoryType::Device
    }
}

/// Safety
/// ------
/// - This works only when `dest` is device memory
#[allow(unused_unsafe)]
pub(super) unsafe fn copy_to_device<T: Copy>(
    dest: &mut impl MemoryMut<Elem = T>,
    src: &impl Memory<Elem = T>,
) {
    assert_ne!(dest.head_addr(), src.head_addr());
    assert_eq!(dest.byte_size(), src.byte_size());

    let dest_ptr = dest.head_addr_mut();
    let src_ptr = src.head_addr();

    // context guard
    let _g = match (dest.try_get_context(), src.try_get_context()) {
        (Some(d_ctx), Some(s_ctx)) => {
            assert_eq!(d_ctx, s_ctx);
            Some(d_ctx.guard_context())
        }
        (Some(ctx), None) => Some(ctx.guard_context()),
        (None, Some(ctx)) => Some(ctx.guard_context()),
        (None, None) => None,
    };

    match src.memory_type() {
        // From host
        MemoryType::Host | MemoryType::Registered | MemoryType::PageLocked => {
            ffi_call!(
                cuMemcpyHtoD_v2,
                dest_ptr as _,
                src_ptr as _,
                dest.byte_size()
            )
            .expect("memcpy from Host to Device failed");
        }
        // From device
        MemoryType::Device => {
            ffi_call!(
                cuMemcpyDtoD_v2,
                dest_ptr as _,
                src_ptr as _,
                dest.byte_size()
            )
            .expect("memcpy from Device to Device failed");
        }
        // From array
        MemoryType::Array => unimplemented!("Array memory is not supported yet"),
    }
}

impl<'ctx, T: Copy> MemoryMut for DeviceMemory<'ctx, T> {
    fn head_addr_mut(&mut self) -> *mut T {
        self.ptr as _
    }
    fn try_as_mut_slice(&mut self) -> Result<&mut [T]> {
        Ok(self.as_mut_slice())
    }
    fn copy_from(&mut self, src: &impl Memory<Elem = Self::Elem>) {
        unsafe { copy_to_device(self, src) }
    }
}

impl<'ctx, T: Copy> Continuous for DeviceMemory<'ctx, T> {
    fn length(&self) -> usize {
        self.size
    }
    fn as_slice(&self) -> &[T] {
        self
    }
}

impl<'ctx, T: Copy> ContinuousMut for DeviceMemory<'ctx, T> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

impl<'ctx, T: Copy> Managed for DeviceMemory<'ctx, T> {}

impl<'ctx, T: Copy> Contexted for DeviceMemory<'ctx, T> {
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
