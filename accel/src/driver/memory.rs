use super::context::*;
use crate::{error::*, ffi_call_unsafe, ffi_new_unsafe};
use cuda::*;
use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
};

pub use cuda::CUmemAttach_flags_enum as AttachFlag;

/// Each variants correspond to the following:
///
/// - Host memory
/// - Device memory
/// - Array memory
/// - Unified device or host memory
pub use cuda::CUmemorytype_enum as MemoryType;

/// Total and Free memory size of the device (in bytes)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryInfo {
    pub free: usize,
    pub total: usize,
}

impl MemoryInfo {
    pub fn get(ctx: &Context) -> Result<Self> {
        ctx.assure_current()?;
        let mut free = 0;
        let mut total = 0;
        ffi_call_unsafe!(
            cuMemGetInfo_v2,
            &mut free as *mut usize,
            &mut total as *mut usize
        )?;
        Ok(MemoryInfo { free, total })
    }
}

/// Trait for CUDA managed memories. It assures
///
/// - can be accessed from both host(CPU) and device(GPU) programs
/// - can be treated as a slice, i.e. a single span of either host or device memory
///
pub trait CudaMemory<T>: Deref<Target = [T]> + DerefMut {
    /// Length of device memory
    fn len(&self) -> usize;
    /// Size of device memory in bytes
    fn byte_size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
    /// Get the unified address of the memory as an immutable pointer
    fn as_ptr(&self) -> *const T;
    /// Get the unified address of the memory as a mutable pointer
    fn as_mut_ptr(&mut self) -> *mut T;
    /// Access as a slice.
    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
    /// Access as a mutable slice.
    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }
}

/// Memory allocated on the device.
pub struct DeviceMemory<'ctx, T> {
    ptr: CUdeviceptr,
    size: usize,
    ctx: &'ctx Context,
    phantom: PhantomData<T>,
}

impl<'ctx, T> Drop for DeviceMemory<'ctx, T> {
    fn drop(&mut self) {
        ffi_call_unsafe!(cuMemFree_v2, self.ptr).expect("Failed to free device memory");
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

impl<'ctx, T> CudaMemory<T> for DeviceMemory<'ctx, T> {
    fn len(&self) -> usize {
        self.size
    }

    fn as_ptr(&self) -> *const T {
        self.ptr as _
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr as _
    }
}

impl<'ctx, T> DeviceMemory<'ctx, T> {
    /// Allocate a new device memory with `size` byte length by [cuMemAllocManaged].
    /// This memory is managed by the unified memory system.
    ///
    /// [cuMemAllocManaged]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467
    pub fn new(ctx: &'ctx Context, size: usize, flag: AttachFlag) -> Result<Self> {
        ctx.assure_current()?;
        let ptr = ffi_new_unsafe!(
            cuMemAllocManaged,
            size * std::mem::size_of::<T>(),
            flag as u32
        )?;
        Ok(DeviceMemory {
            ptr,
            size,
            ctx,
            phantom: PhantomData,
        })
    }

    fn get_attr<Attr>(&self, attr: CUpointer_attribute) -> Result<Attr> {
        self.ctx.assure_current()?;
        let ty = MaybeUninit::uninit();
        ffi_call_unsafe!(cuPointerGetAttribute, ty.as_ptr() as *mut _, attr, self.ptr)?;
        let ty = unsafe { ty.assume_init() };
        Ok(ty)
    }

    /// Unique ID of the memory
    pub fn buffer_id(&self) -> Result<u64> {
        self.get_attr(CUpointer_attribute::CU_POINTER_ATTRIBUTE_BUFFER_ID)
    }

    /// Check if the memory is managed by the unified memory system
    pub fn assure_managed(&self) -> Result<()> {
        if self.get_attr::<bool>(CUpointer_attribute::CU_POINTER_ATTRIBUTE_IS_MANAGED)? {
            Ok(())
        } else {
            Err(AccelError::DeviceMemoryIsNotManaged)
        }
    }

    pub fn memory_type(&self) -> Result<MemoryType> {
        self.get_attr(CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE)
    }

    /// Check if this pointer is in a valid address range that is mapped to a backing allocation.
    /// This will always returns true
    pub fn is_mapped(&self) -> Result<bool> {
        self.get_attr(CUpointer_attribute::CU_POINTER_ATTRIBUTE_MAPPED)
    }
}

#[cfg(test)]
mod device_memory_tests {
    use super::super::device::*;
    use super::*;

    #[test]
    fn info() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let mem_info = MemoryInfo::get(&ctx)?;
        dbg!(&mem_info);
        assert!(mem_info.free > 0);
        assert!(mem_info.total > mem_info.free);
        Ok(())
    }

    #[test]
    fn managed() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let mut mem = DeviceMemory::<i32>::new(&ctx, 12, AttachFlag::CU_MEM_ATTACH_GLOBAL)?;
        assert_eq!(mem.len(), 12);
        assert_eq!(mem.byte_size(), 12 * 4 /* size of i32 */);
        let sl = mem.as_mut_slice();
        sl[0] = 3;
        Ok(())
    }

    #[test]
    fn pointer_attributes() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;

        let mem1 = DeviceMemory::<i32>::new(&ctx, 12, AttachFlag::CU_MEM_ATTACH_GLOBAL)?;
        assert_eq!(mem1.memory_type()?, MemoryType::CU_MEMORYTYPE_DEVICE);
        assert!(mem1.assure_managed().is_ok());
        assert!(mem1.is_mapped()?);

        // Buffer id of two different memory must be different
        let mem2 = DeviceMemory::<i32>::new(&ctx, 12, AttachFlag::CU_MEM_ATTACH_GLOBAL)?;
        assert_ne!(mem1.buffer_id()?, mem2.buffer_id()?);
        Ok(())
    }
}

pub struct PageLockedMemory<T> {
    ptr: *mut T,
    size: usize,
}

impl<T> Drop for PageLockedMemory<T> {
    fn drop(&mut self) {
        ffi_call_unsafe!(cuMemFreeHost, self.ptr as *mut _)
            .expect("Cannot free page-locked memory");
    }
}

impl<T> Deref for PageLockedMemory<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for PageLockedMemory<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> CudaMemory<T> for PageLockedMemory<T> {
    fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    fn len(&self) -> usize {
        self.size
    }
}

impl<T> PageLockedMemory<T> {
    pub fn new(size: usize) -> Self {
        let ptr = ffi_new_unsafe!(cuMemAllocHost_v2, size * std::mem::size_of::<T>())
            .expect("Cannot allocate page-locked memory");
        Self {
            ptr: ptr as *mut T,
            size,
        }
    }
}

#[cfg(test)]
mod page_locaked_memory_tests {
    use super::*;

    #[should_panic]
    #[test]
    fn zero_new() {
        let _a: PageLockedMemory<i32> = PageLockedMemory::new(0);
    }
}
