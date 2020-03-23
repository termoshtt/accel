use super::context::*;
use crate::{error::*, ffi_call, ffi_call_unsafe, ffi_new_unsafe};
use cuda::*;
use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
};

use cuda::CUmemAttach_flags_enum as AttachFlag;

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

    /// Unique identifier of the memory
    ///
    /// ```
    /// # use ::accel::driver::{device::*, context::*, memory::*};
    /// # let device = Device::nth(0).unwrap();
    /// # let ctx = device.create_context_auto().unwrap();
    /// let mem1 = DeviceMemory::<i32>::new(&ctx, 12);
    /// let mem2 = DeviceMemory::<i32>::new(&ctx, 12);
    /// assert_ne!(mem1.id(), mem2.id());
    /// ```
    fn id(&self) -> u64 {
        get_attr(
            self.as_ptr(),
            CUpointer_attribute::CU_POINTER_ATTRIBUTE_BUFFER_ID,
        )
    }

    /// Memory Type
    fn memory_type(&self) -> MemoryType {
        get_attr(
            self.as_ptr(),
            CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        )
    }
}

// Typed wrapper of cuPointerGetAttribute
fn get_attr<T, Attr>(ptr: *const T, attr: CUpointer_attribute) -> Attr {
    let data = MaybeUninit::uninit();
    unsafe {
        ffi_call!(
            cuPointerGetAttribute,
            data.as_ptr() as *mut _,
            attr,
            ptr as CUdeviceptr
        )
        .expect("Cannot get pointer attributes");
        data.assume_init()
    }
}

/// Memory allocated on the device.
pub struct DeviceMemory<T> {
    ptr: CUdeviceptr,
    size: usize,
    phantom: PhantomData<T>,
}

impl<T> Drop for DeviceMemory<T> {
    fn drop(&mut self) {
        ffi_call_unsafe!(cuMemFree_v2, self.ptr).expect("Failed to free device memory");
    }
}

impl<T> Deref for DeviceMemory<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for DeviceMemory<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> CudaMemory<T> for DeviceMemory<T> {
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

impl<T> DeviceMemory<T> {
    /// Allocate a new device memory with `size` byte length by [cuMemAllocManaged].
    /// This memory is managed by the unified memory system.
    ///
    /// [cuMemAllocManaged]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467
    pub fn new(ctx: &Context, size: usize) -> Self {
        ctx.assure_current()
            .expect("DeviceMemory::new requires valid and current context");
        let ptr = ffi_new_unsafe!(
            cuMemAllocManaged,
            size * std::mem::size_of::<T>(),
            AttachFlag::CU_MEM_ATTACH_GLOBAL as u32
        )
        .expect("Cannot allocate device memory");
        DeviceMemory {
            ptr,
            size,
            phantom: PhantomData,
        }
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
mod tests {
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

    #[should_panic(expected = "Cannot allocate")]
    #[test]
    fn page_locked_new_zero() {
        let _a = PageLockedMemory::<i32>::new(0);
    }

    #[should_panic(expected = "Cannot allocate")]
    #[test]
    fn device_new_zero() {
        let device = Device::nth(0).unwrap();
        let ctx = device.create_context_auto().unwrap();
        let _a = DeviceMemory::<i32>::new(&ctx, 0);
    }

    #[test]
    fn device() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let mut mem = DeviceMemory::<i32>::new(&ctx, 12);
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

        let mem1 = DeviceMemory::<i32>::new(&ctx, 12);
        assert_eq!(mem1.memory_type(), MemoryType::CU_MEMORYTYPE_DEVICE);

        // Buffer id of two different memory must be different
        let mem2 = DeviceMemory::<i32>::new(&ctx, 12);
        assert_ne!(mem1.id(), mem2.id());
        Ok(())
    }
}
