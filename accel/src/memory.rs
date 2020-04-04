use super::device::*;
use crate::{ffi_call, ffi_new};
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
struct MemoryInfo {
    free: usize,
    total: usize,
}

impl MemoryInfo {
    fn get(ctx: &Context) -> Self {
        let _gurad = ctx.guard_context();
        let mut free = 0;
        let mut total = 0;
        ffi_call!(
            cuMemGetInfo_v2,
            &mut free as *mut usize,
            &mut total as *mut usize
        )
        .expect("Cannot get memory info");
        MemoryInfo { free, total }
    }
}

/// Get total memory size in bytes of the current device
///
/// Panic
/// ------
/// - when given context is not current
pub fn total_memory(ctx: &Context) -> usize {
    MemoryInfo::get(ctx).total
}

/// Get free memory size in bytes of the current device
///
/// Panic
/// ------
/// - when given context is not current
pub fn free_memory(ctx: &Context) -> usize {
    MemoryInfo::get(ctx).free
}

/// Trait for CUDA managed memories. It assures
///
/// - can be accessed from both host (CPU) and device (GPU) programs
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
    /// # use ::accel::{device::*, memory::*};
    /// # let device = Device::nth(0).unwrap();
    /// # let ctx = device.create_context();
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
    ///
    /// ```
    /// # use ::accel::{device::*, memory::*};
    /// # let device = Device::nth(0).unwrap();
    /// # let ctx = device.create_context();
    /// let dev = DeviceMemory::<i32>::new(&ctx, 12);
    /// assert_eq!(dev.memory_type(), MemoryType::CU_MEMORYTYPE_DEVICE);
    /// let host = PageLockedMemory::<i32>::new(&ctx, 12);
    /// assert_eq!(host.memory_type(), MemoryType::CU_MEMORYTYPE_HOST);
    /// ```
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
    ffi_call!(
        cuPointerGetAttribute,
        data.as_ptr() as *mut _,
        attr,
        ptr as CUdeviceptr
    )
    .expect("Cannot get pointer attributes");
    unsafe { data.assume_init() }
}

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

/// Memory allocated as page-locked
pub struct PageLockedMemory<'ctx, T> {
    ptr: *mut T,
    size: usize,
    context: &'ctx Context,
}

impl<'ctx, T> Drop for PageLockedMemory<'ctx, T> {
    fn drop(&mut self) {
        if let Err(e) = ffi_call!(cuMemFreeHost, self.ptr as *mut _) {
            log::error!("Cannot free page-locked memory: {:?}", e);
        }
    }
}

impl<'ctx, T> Deref for PageLockedMemory<'ctx, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'ctx, T> DerefMut for PageLockedMemory<'ctx, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'ctx, T> Contexted for PageLockedMemory<'ctx, T> {
    fn get_context(&self) -> &Context {
        &self.context
    }
}

impl<'ctx, T> CudaMemory<T> for PageLockedMemory<'ctx, T> {
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

impl<'ctx, T> PageLockedMemory<'ctx, T> {
    /// Allocate host memory as page-locked.
    ///
    /// Allocating excessive amounts of pinned memory may degrade system performance,
    /// since it reduces the amount of memory available to the system for paging.
    /// As a result, this function is best used sparingly to allocate staging areas for data exchange between host and device.
    ///
    /// See also [cuMemAllocHost].
    ///
    /// [cuMemAllocHost]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0
    ///
    /// Panic
    /// ------
    /// - when memory allocation failed includeing `size == 0` case
    ///
    pub fn new(context: &'ctx Context, size: usize) -> Self {
        assert!(size > 0, "Zero-sized malloc is forbidden");
        let _g = context.guard_context();
        let ptr = ffi_new!(cuMemAllocHost_v2, size * std::mem::size_of::<T>())
            .expect("Cannot allocate page-locked memory");
        Self {
            ptr: ptr as *mut T,
            size,
            context,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::error::*;
    use super::*;

    #[test]
    fn info() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let mem_info = MemoryInfo::get(&ctx);
        dbg!(&mem_info);
        assert!(mem_info.free > 0);
        assert!(mem_info.total > mem_info.free);
        Ok(())
    }

    #[test]
    fn page_locked() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let mut mem = PageLockedMemory::<i32>::new(&ctx, 12);
        assert_eq!(mem.len(), 12);
        assert_eq!(mem.byte_size(), 12 * 4 /* size of i32 */);
        let sl = mem.as_mut_slice();
        sl[0] = 3;
        Ok(())
    }

    #[should_panic(expected = "Zero-sized malloc is forbidden")]
    #[test]
    fn page_locked_new_zero() {
        let device = Device::nth(0).unwrap();
        let ctx = device.create_context();
        let _a = PageLockedMemory::<i32>::new(&ctx, 0);
    }

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
