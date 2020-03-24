use super::context::*;
use crate::{error::*, ffi_call, ffi_call_unsafe, ffi_new_unsafe};
use cuda::*;
use derive_new::new;
use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
};

use cuda::CUmemAttach_flags_enum as AttachFlag;

pub use cuda::CUarray_format as ArrayFormat;

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
        ctx.assure_current().expect("Non-current context");
        let mut free = 0;
        let mut total = 0;
        ffi_call_unsafe!(
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
    ///
    /// ```
    /// # use ::accel::driver::{device::*, context::*, memory::*};
    /// # let device = Device::nth(0).unwrap();
    /// # let ctx = device.create_context_auto().unwrap();
    /// let dev = DeviceMemory::<i32>::new(&ctx, 12);
    /// assert_eq!(dev.memory_type(), MemoryType::CU_MEMORYTYPE_DEVICE);
    /// let host = PageLockedMemory::<i32>::new(12);
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
    /// Panic
    /// ------
    /// - when given context is not current
    /// - allocation failed including `size == 0` case
    ///
    /// [cuMemAllocManaged]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467
    ///
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

/// Memory allocated as page-locked
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
    pub fn new(size: usize) -> Self {
        let ptr = ffi_new_unsafe!(cuMemAllocHost_v2, size * std::mem::size_of::<T>())
            .expect("Cannot allocate page-locked memory");
        Self {
            ptr: ptr as *mut T,
            size,
        }
    }
}

#[derive(Debug)]
pub struct Array<T, Dim> {
    array: CUarray,
    dim: Dim,
    num_channels: u32,
    phantom: PhantomData<T>,
}

impl<T, Dim> Drop for Array<T, Dim> {
    fn drop(&mut self) {
        ffi_call_unsafe!(cuArrayDestroy, self.array).expect("Failed to cleanup array");
    }
}

impl<T: Scalar, Dim: Dimension> Array<T, Dim> {
    /// Create a new array on the device.
    ///
    /// - `num_channels` specifies the number of packed components per CUDA array element; it may be 1, 2, or 4;
    ///   - e.g. `T=f32` and `num_channels == 2`, then the size of an element is 64bit as packed two 32bit float values
    ///
    /// Panic
    /// -----
    /// - when allocation failed
    ///
    pub fn new(dim: impl Into<Dim>, num_channels: u32) -> Self {
        let dim = dim.into();
        let desc = dim.as_descriptor::<T>(num_channels);
        let array = ffi_new_unsafe!(cuArray3DCreate_v2, &desc).expect("Cannot create a new array");
        Array {
            array,
            dim,
            num_channels,
            phantom: PhantomData,
        }
    }

    pub fn dim(&self) -> &Dim {
        &self.dim
    }

    pub fn element_size(&self) -> usize {
        std::mem::size_of::<T>() * self.num_channels as usize
    }

    pub fn len(&self) -> usize {
        self.dim.len()
    }

    pub fn num_channels(&self) -> u32 {
        self.num_channels
    }
}

pub trait Dimension {
    /// `num_channels` specifies the number of packed components per CUDA array element; it may be 1, 2, or 4;
    fn as_descriptor<T: Scalar>(&self, num_channels: u32) -> CUDA_ARRAY3D_DESCRIPTOR;
    /// Number of elements
    fn len(&self) -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix1 {
    pub width: usize,
}

impl Dimension for Ix1 {
    fn as_descriptor<T: Scalar>(&self, num_channels: u32) -> CUDA_ARRAY3D_DESCRIPTOR {
        CUDA_ARRAY3D_DESCRIPTOR {
            Width: self.width,
            Height: 0,
            Depth: 0,
            NumChannels: num_channels,
            Flags: ArrayFlag::empty().bits(),
            Format: T::format(),
        }
    }
    fn len(&self) -> usize {
        self.width
    }
}

#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix2 {
    pub width: usize,
    pub hight: usize,
}

impl Dimension for Ix2 {
    fn as_descriptor<T: Scalar>(&self, num_channels: u32) -> CUDA_ARRAY3D_DESCRIPTOR {
        CUDA_ARRAY3D_DESCRIPTOR {
            Width: self.width,
            Height: self.hight,
            Depth: 0,
            NumChannels: num_channels,
            Flags: ArrayFlag::empty().bits(),
            Format: T::format(),
        }
    }
    fn len(&self) -> usize {
        self.width * self.hight
    }
}

#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix3 {
    pub width: usize,
    pub hight: usize,
    pub depth: usize,
}

impl Dimension for Ix3 {
    fn as_descriptor<T: Scalar>(&self, num_channels: u32) -> CUDA_ARRAY3D_DESCRIPTOR {
        CUDA_ARRAY3D_DESCRIPTOR {
            Width: self.width,
            Height: self.hight,
            Depth: self.depth,
            NumChannels: num_channels,
            Flags: ArrayFlag::empty().bits(),
            Format: T::format(),
        }
    }
    fn len(&self) -> usize {
        self.width * self.hight * self.depth
    }
}

#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix1Layered {
    pub width: usize,
    pub depth: usize,
}

impl Dimension for Ix1Layered {
    fn as_descriptor<T: Scalar>(&self, num_channels: u32) -> CUDA_ARRAY3D_DESCRIPTOR {
        CUDA_ARRAY3D_DESCRIPTOR {
            Width: self.width,
            Height: 0,
            Depth: self.depth,
            NumChannels: num_channels,
            Flags: ArrayFlag::LAYERED.bits(),
            Format: T::format(),
        }
    }
    fn len(&self) -> usize {
        self.width * self.depth
    }
}

#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix2Layered {
    pub width: usize,
    pub hight: usize,
    pub depth: usize,
}

impl Dimension for Ix2Layered {
    fn as_descriptor<T: Scalar>(&self, num_channels: u32) -> CUDA_ARRAY3D_DESCRIPTOR {
        CUDA_ARRAY3D_DESCRIPTOR {
            Width: self.width,
            Height: self.hight,
            Depth: self.depth,
            NumChannels: num_channels,
            Flags: ArrayFlag::LAYERED.bits(),
            Format: T::format(),
        }
    }
    fn len(&self) -> usize {
        self.width * self.hight * self.depth
    }
}

pub trait Scalar {
    fn format() -> ArrayFormat;
}

macro_rules! impl_array_scalar {
    ($scalar:ty, $format:ident) => {
        impl Scalar for $scalar {
            fn format() -> ArrayFormat {
                ArrayFormat::$format
            }
        }
    };
}

impl_array_scalar!(u8, CU_AD_FORMAT_UNSIGNED_INT8);
impl_array_scalar!(u16, CU_AD_FORMAT_UNSIGNED_INT16);
impl_array_scalar!(u32, CU_AD_FORMAT_UNSIGNED_INT32);
impl_array_scalar!(i8, CU_AD_FORMAT_SIGNED_INT8);
impl_array_scalar!(i16, CU_AD_FORMAT_SIGNED_INT16);
impl_array_scalar!(i32, CU_AD_FORMAT_SIGNED_INT32);
// FIXME f16 is not supported yet
impl_array_scalar!(f32, CU_AD_FORMAT_FLOAT);

bitflags::bitflags! {
    struct ArrayFlag: u32 {
        /// If set, the CUDA array is a collection of layers, where each layer is either a 1D or a 2D array and the Depth member of CUDA_ARRAY3D_DESCRIPTOR specifies the number of layers, not the depth of a 3D array.
        const LAYERED = 0x01;
        /// This flag must be set in order to bind a surface reference to the CUDA array
        const SURFACE_LDST = 0x02;
        /// If set, the CUDA array is a collection of six 2D arrays, representing faces of a cube. The width of such a CUDA array must be equal to its height, and Depth must be six. If CUDA_ARRAY3D_LAYERED flag is also set, then the CUDA array is a collection of cubemaps and Depth must be a multiple of six.
        const CUBEMAP = 0x04;
        /// This flag must be set in order to perform texture gather operations on a CUDA array.
        const TEXTURE_GATHER = 0x08;
        /// This flag if set indicates that the CUDA array is a DEPTH_TEXTURE.
        const DEPTH_TEXTURE = 0x10;
        /// This flag indicates that the CUDA array may be bound as a color target in an external graphics API
        const COLOR_ATTACHMENT = 0x20;
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
        let mem_info = MemoryInfo::get(&ctx);
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
}
