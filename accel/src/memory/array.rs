//! CUDA [Array] and [Texture], [Surface] Objects
//!
//! [Array]:   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-arrays
//! [Texture]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TEXOBJECT.html#group__CUDA__TEXOBJECT
//! [Surface]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__SURFOBJECT.html#group__CUDA__SURFOBJECT

use crate::{contexted_call, contexted_new, device::Contexted, *};
use cuda::*;
use std::{marker::PhantomData, sync::Arc};

pub use cuda::CUDA_ARRAY3D_DESCRIPTOR as Descriptor;

#[derive(Debug)]
pub struct Array<T, Dim> {
    array: CUarray,
    dim: Dim,
    ctx: Arc<Context>,
    phantom: PhantomData<T>,
}

impl<T, Dim> Drop for Array<T, Dim> {
    fn drop(&mut self) {
        if let Err(e) = unsafe { contexted_call!(self, cuArrayDestroy, self.array) } {
            log::error!("Failed to cleanup array: {:?}", e);
        }
    }
}

impl<T: Scalar, Dim: Dimension> Array<T, Dim> {
    /// Get dimension
    pub fn dim(&self) -> &Dim {
        &self.dim
    }
}

impl<T: Scalar, Dim: Dimension> Memory for Array<T, Dim> {
    type Elem = T;
    fn head_addr(&self) -> *const T {
        self.array as _
    }
    fn head_addr_mut(&mut self) -> *mut T {
        self.array as _
    }

    fn num_elem(&self) -> usize {
        self.dim.len()
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::Array
    }

    fn try_as_slice(&self) -> Option<&[Self::Elem]> {
        None
    }

    fn try_as_mut_slice(&mut self) -> Option<&mut [Self::Elem]> {
        None
    }

    fn try_get_context(&self) -> Option<Arc<Context>> {
        Some(self.get_context())
    }
}

impl<T: Scalar, Dim: Dimension> Memcpy<PageLockedMemory<T>> for Array<T, Dim> {
    fn copy_from(&mut self, src: &PageLockedMemory<T>) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        let desc = self.dim.as_descriptor::<T>();
        let param = CUDA_MEMCPY3D {
            srcMemoryType: CUmemorytype_enum::CU_MEMORYTYPE_HOST,
            srcHost: src.as_ptr() as *mut _,
            srcPitch: desc.Width * T::size_of(),
            srcHeight: desc.Height,

            dstMemoryType: CUmemorytype_enum::CU_MEMORYTYPE_ARRAY,
            dstArray: self.array,

            WidthInBytes: desc.Width * T::size_of() * desc.NumChannels as usize,
            Height: desc.Height,
            Depth: desc.Depth,

            ..Default::default()
        };
        unsafe { contexted_call!(self, cuMemcpy3D_v2, &param) }
            .expect("memcpy from Array to page-locked host memory failed");
    }

    fn copy_to(&self, dest: &mut PageLockedMemory<T>) {
        assert_ne!(self.head_addr(), dest.head_addr());
        assert_eq!(self.num_elem(), dest.num_elem());
        todo!()
    }
}

// use default impl
impl<T: Scalar, Dim: Dimension> Memcpy<Array<T, Dim>> for PageLockedMemory<T> {}

impl<T: Scalar, Dim: Dimension> Memcpy<DeviceMemory<T>> for Array<T, Dim> {
    fn copy_from(&mut self, src: &DeviceMemory<T>) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        todo!()
    }

    fn copy_to(&self, dest: &mut DeviceMemory<T>) {
        assert_ne!(self.head_addr(), dest.head_addr());
        assert_eq!(self.num_elem(), dest.num_elem());
        todo!()
    }
}

// use default impl
impl<T: Scalar, Dim: Dimension> Memcpy<Array<T, Dim>> for DeviceMemory<T> {}

impl<T: Scalar, Dim: Dimension> Memset for Array<T, Dim> {
    fn set(&mut self, value: Self::Elem) {
        // FIXME CUDA does not have memcpy for array. This is easy but too expensive alternative way
        let src = PageLockedMemory::from_elem(self.get_context(), self.dim.len(), value);
        self.copy_from(&src);
    }
}

impl<T, Dim> Contexted for Array<T, Dim> {
    fn get_context(&self) -> Arc<Context> {
        self.ctx.clone()
    }
}

impl<T: Scalar, Dim: Dimension> Allocatable for Array<T, Dim> {
    type Shape = Dim;
    unsafe fn uninitialized(ctx: Arc<Context>, dim: Dim) -> Self {
        let desc = dim.as_descriptor::<T>();
        let array =
            contexted_new!(&ctx, cuArray3DCreate_v2, &desc).expect("Cannot create a new array");
        Array {
            array,
            dim,
            ctx,
            phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{device::*, error::*};

    #[test]
    fn new_1d() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _array1: Array<f32, Ix1> = Array::zeros(ctx.clone(), 10.into());
        let _array2: Array<f32, Ix1> = Array::zeros(ctx.clone(), (10,).into());
        Ok(())
    }

    #[test]
    fn new_2d() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _array: Array<f32, Ix2> = Array::zeros(ctx, (10, 12).into());
        Ok(())
    }

    #[test]
    fn new_3d() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _array: Array<f32, Ix3> = Array::zeros(ctx, (10, 12, 8).into());
        Ok(())
    }

    #[test]
    fn new_1d_layered() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _array: Array<f32, Ix1Layered> = Array::zeros(ctx, (10, 12).into());
        Ok(())
    }

    #[test]
    fn new_2d_layered() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _array: Array<f32, Ix2Layered> = Array::zeros(ctx, (10, 12, 8).into());
        Ok(())
    }
}
