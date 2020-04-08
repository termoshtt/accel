//! Memory management
//!
//! Unified address
//! ---------------
//!
//! - All memories are mapped into a single 64bit memory space
//!
//! Memory Types
//! ------------
//!
//! |name                     | where exists | From Host | From Device | Description                                                               |
//! |:------------------------|:------------:|:---------:|:-----------:|:--------------------------------------------------------------------------|
//! | (usual) Host memory     | Host         | ✓         |  -          | allocated by usual manner, e.g. `vec![0; n]`                              |
//! | registered Host memory  | Host         | ✓         |  ✓          | allocated by usual manner, and registered into CUDA unified memory system |
//! | Page-locked Host memory | Host         | ✓         |  ✓          | OS memory paging feature is disabled for accelarating memory transfer     |
//! | Device memory           | Device       | ✓         |  ✓          | allocated on device as a single span                                      |
//! | Array                   | Device       | ✓         |  ✓          | properly aligned memory on device for using Texture and Surface memory    |
//!

pub mod array;
pub mod device;
pub mod host;
mod info;

pub use array::*;
pub use device::*;
pub use host::*;
pub use info::*;

use crate::{error::*, ffi_call};
use cuda::*;
use std::{
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
};

/// Each variants correspond to the following:
///
/// - Host memory
/// - Device memory
/// - Array memory
/// - Unified device or host memory
pub use cuda::CUmemorytype_enum as MemoryType;

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
    /// # use ::accel::*;
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
        .unwrap()
    }

    /// Memory Type
    ///
    /// ```
    /// # use ::accel::*;
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
        .unwrap()
    }
}

// Typed wrapper of cuPointerGetAttribute
fn get_attr<T, Attr>(ptr: *const T, attr: CUpointer_attribute) -> Result<Attr> {
    let data = MaybeUninit::uninit();
    ffi_call!(
        cuPointerGetAttribute,
        data.as_ptr() as *mut _,
        attr,
        ptr as CUdeviceptr
    )?;
    unsafe { data.assume_init() }
}

pub fn memory_type<T>(ptr: *const T) -> Result<MemoryType> {
    get_attr(ptr, CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE)
}

pub fn buffer_id<T>(ptr: *const T) -> Result<u64> {
    get_attr(ptr, CUpointer_attribute::CU_POINTER_ATTRIBUTE_BUFFER_ID)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::*;

    #[test]
    fn get_attr_host_memory() -> Result<()> {
        let device = Device::nth(0)?;
        let _ctx = device.create_context();

        let a = vec![0_u32; 12];
        let attr = memory_type(a.as_ptr());
        dbg!(attr);
        let id = buffer_id(a.as_ptr());
        dbg!(id);
        panic!("!!");
    }
}
