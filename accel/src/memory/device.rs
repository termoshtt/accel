//! Device and Host memory handlers

use super::*;
use crate::*;
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
        if let Err(e) = unsafe { contexted_call!(self, cuMemFree_v2, self.ptr) } {
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

impl<'ctx, T: Scalar> Memory for DeviceMemory<'ctx, T> {
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

    fn try_as_slice(&self) -> Option<&[T]> {
        Some(self.as_slice())
    }

    fn try_as_mut_slice(&mut self) -> Option<&mut [T]> {
        Some(self.as_mut_slice())
    }

    fn try_get_context(&self) -> Option<&Context> {
        Some(self.get_context())
    }
}

/// Safety
/// ------
/// - This works only when `mem` is device memory
pub(super) unsafe fn memset_device<T, Mem>(mem: &mut Mem, value: T) -> error::Result<()>
where
    T: Scalar,
    Mem: Continuous<Elem = T> + ?Sized,
{
    assert_eq!(mem.memory_type(), MemoryType::Device);
    match T::size_of() {
        bytes @ 1 | bytes @ 2 | bytes @ 4 => {
            let ptr = mem.head_addr_mut() as _;
            let size = mem.num_elem();
            let ctx = mem.try_get_context().unwrap();
            match bytes {
                1 => {
                    let value = value.to_le_u8().unwrap();
                    contexted_call!(ctx, cuMemsetD8_v2, ptr, value, size)?;
                }
                2 => {
                    let value = value.to_le_u16().unwrap();
                    contexted_call!(ctx, cuMemsetD16_v2, ptr, value, size)?;
                }
                4 => {
                    let value = value.to_le_u32().unwrap();
                    contexted_call!(ctx, cuMemsetD32_v2, ptr, value, size)?;
                }
                _ => unreachable!(),
            }
        }
        _ => mem.as_mut_slice().iter_mut().for_each(|v| *v = value),
    }
    Ok(())
}

/// Safety
/// ------
/// - This works only when `dest` is device memory
pub(super) unsafe fn copy_to_device<T: Scalar, Dest, Src>(dest: &mut Dest, src: &Src)
where
    Dest: Memory<Elem = T> + ?Sized,
    Src: Memory<Elem = T> + ?Sized,
{
    assert_eq!(dest.memory_type(), MemoryType::Device);
    assert_ne!(dest.head_addr(), src.head_addr());
    assert_eq!(dest.num_elem(), src.num_elem());

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
                dest.num_elem() * std::mem::size_of::<T>()
            )
            .expect("memcpy from Host to Device failed");
        }
        // From device
        MemoryType::Device => {
            ffi_call!(
                cuMemcpyDtoD_v2,
                dest_ptr as _,
                src_ptr as _,
                dest.num_elem() * std::mem::size_of::<T>()
            )
            .expect("memcpy from Device to Device failed");
        }
        // From array
        MemoryType::Array => unimplemented!("Array memory is not supported yet"),
    }
}

impl<'ctx, T, Source> Memcpy<Source> for DeviceMemory<'ctx, T>
where
    T: Scalar,
    Source: Memory<Elem = T> + ?Sized,
{
    fn copy_from(&mut self, src: &Source) {
        unsafe { copy_to_device(self, src) }
    }
}

impl<'ctx, T: Scalar> Memset for DeviceMemory<'ctx, T> {
    fn set(&mut self, value: Self::Elem) {
        unsafe { memset_device(self, value).expect("memset failed") };
    }
}

impl<'ctx, T: Scalar> Continuous for DeviceMemory<'ctx, T> {
    fn as_slice(&self) -> &[T] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

impl<'ctx, T: Scalar> Managed for DeviceMemory<'ctx, T> {}

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
        let ptr = unsafe {
            contexted_new!(
                context,
                cuMemAllocManaged,
                size * std::mem::size_of::<T>(),
                AttachFlag::CU_MEM_ATTACH_GLOBAL as u32
            )
        }
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
        let _a = DeviceMemory::<i32>::new(&ctx, 0);
    }
}
