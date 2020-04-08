//! Memory management
//!
//! Unified address
//! ---------------
//!
//! - All memories are mapped into a single 64bit memory space
//! - We can get where the pointed memory exists from its value.
//!
//! Memory Types
//! ------------
//!
//! |name                      | where exists | From Host | From Device | As slice | Description                                                            |
//! |:-------------------------|:------------:|:---------:|:-----------:|:--------:|:-----------------------------------------------------------------------|
//! | (usual) Host memory      | Host         | ✓         |  -          |  ✓       | allocated by usual manner, e.g. `vec![0; n]`                           |
//! | [Registered Host memory] | Host         | ✓         |  ✓          |  ✓       | A host memory registered into CUDA memory management system            |
//! | [Page-locked Host memory]| Host         | ✓         |  ✓          |  ✓       | OS memory paging is disabled for accelerating memory transfer          |
//! | [Device memory]          | Device       | ✓         |  ✓          |  ✓       | allocated on device as a single span                                   |
//! | [Array]                  | Device       | ✓         |  ✓          |  -       | properly aligned memory on device for using Texture and Surface memory |
//!
//! [Registered Host memory]:  ./host/struct.RegisterdMemory.html
//! [Page-locked Host memory]: ./host/struct.PageLockedMemory.html
//! [Device memory]:           ./device/struct.DeviceMemory.html
//! [Array]:                   ./array/struct.Array.html
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
pub use cuda::CUmemorytype_enum;

pub enum MemoryType {
    Host,
    Registered,
    PageLocked,
    Device,
    Array,
}

impl Into<MemoryType> for CUmemorytype_enum {
    fn into(self) -> MemoryType {
        match self {
            CUmemorytype_enum::CU_MEMORYTYPE_HOST => MemoryType::PageLocked,
            CUmemorytype_enum::CU_MEMORYTYPE_DEVICE => MemoryType::Device,
            CUmemorytype_enum::CU_MEMORYTYPE_ARRAY => MemoryType::Array,
            CUmemorytype_enum::CU_MEMORYTYPE_UNIFIED => MemoryType::Registered,
        }
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

fn memory_type<T>(ptr: *const T) -> MemoryType {
    match get_attr(ptr, CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE) {
        Ok(CUmemorytype_enum::CU_MEMORYTYPE_HOST) => MemoryType::PageLocked,
        Ok(CUmemorytype_enum::CU_MEMORYTYPE_DEVICE) => MemoryType::Device,
        Ok(CUmemorytype_enum::CU_MEMORYTYPE_ARRAY) => MemoryType::Array,
        Ok(CUmemorytype_enum::CU_MEMORYTYPE_UNIFIED) => MemoryType::Registered,
        Err(_) => MemoryType::Host,
    }
}

fn buffer_id<T>(ptr: *const T) -> Result<u64> {
    get_attr(ptr, CUpointer_attribute::CU_POINTER_ATTRIBUTE_BUFFER_ID)
}

/// Has unique head address and allocated size.
pub trait Memory {
    type Elem;
    fn head_addr(&self) -> *const Self::Elem;
    fn byte_size(&self) -> usize;
    fn memory_type(&self) -> MemoryType {
        memory_type(self.head_addr())
    }
}

/// Has unique head address and allocated size.
pub trait MemoryMut: Memory {
    fn head_addr_mut(&mut self) -> *mut Self::Elem;
}

impl<'a, T> Memory for &'a [T] {
    type Elem = T;
    fn head_addr(&self) -> *const T {
        self.as_ptr()
    }
    fn byte_size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
}

impl<'a, T> Memory for &'a mut [T] {
    type Elem = T;
    fn head_addr(&self) -> *const T {
        self.as_ptr()
    }
    fn byte_size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
}

impl<'a, T> MemoryMut for &'a mut [T] {
    fn head_addr_mut(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

/// Has 1D index in addition to [Memory] trait.
pub trait Continuous: Memory {
    fn length(&self) -> usize;
    fn as_slice(&self) -> &[Self::Elem];
}

/// Has 1D index in addition to [Memory] trait.
pub trait ContinuousMut: Continuous {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
}

impl<'a, T> Continuous for &'a [T] {
    fn length(&self) -> usize {
        self.len()
    }
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

impl<'a, T> Continuous for &'a mut [T] {
    fn length(&self) -> usize {
        self.len()
    }
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

impl<'a, T> ContinuousMut for &'a mut [T] {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}

/// Is managed under the CUDA unified memory management systems in addition to [Memory] trait.
pub trait Managed: Memory {
    fn buffer_id(&self) -> u64 {
        buffer_id(self.head_addr()).expect("Not managed by CUDA")
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_for_slice() -> Result<()> {
        let a = vec![0_u32; 12];
        assert!(matches!(a.as_slice().memory_type(), MemoryType::Host));
        assert_eq!(a.as_slice().length(), 12);
        assert_eq!(a.as_slice().byte_size(), 12 * 4 /* u32 */);
        Ok(())
    }
}
