//! CUDA [Array] and [Texture], [Surface] Objects
//!
//! [Array]:   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-arrays
//! [Texture]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TEXOBJECT.html#group__CUDA__TEXOBJECT
//! [Surface]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__SURFOBJECT.html#group__CUDA__SURFOBJECT

use crate::{contexted_call, contexted_new, device::Contexted, error::Result, *};
use cuda::*;
use futures::future::BoxFuture;
use num_traits::ToPrimitive;
use std::marker::PhantomData;

pub use cuda::CUDA_ARRAY3D_DESCRIPTOR as Descriptor;

#[derive(Debug, Contexted)]
pub struct Array<T, Dim> {
    array: CUarray,
    dim: Dim,
    context: Context,
    phantom: PhantomData<T>,
}

unsafe impl<T, Dim> Send for Array<T, Dim> {}
unsafe impl<T, Dim> Sync for Array<T, Dim> {}

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
}

fn memcpy3d_param_h2a<T: Scalar, Dim: Dimension>(
    src: &[T],
    dst: &mut Array<T, Dim>,
) -> CUDA_MEMCPY3D {
    let dim = dst.dim;
    CUDA_MEMCPY3D {
        srcMemoryType: CUmemorytype_enum::CU_MEMORYTYPE_UNIFIED,
        srcDevice: src.as_ptr() as CUdeviceptr,

        dstMemoryType: CUmemorytype_enum::CU_MEMORYTYPE_ARRAY,
        dstArray: dst.array,

        WidthInBytes: dim.width() * T::size_of() * dim.num_channels().to_usize().unwrap(),
        Height: dim.height(),
        Depth: dim.depth(),

        ..Default::default()
    }
}

impl<T: Scalar, Dim: Dimension> Memcpy<[T]> for Array<T, Dim> {
    fn copy_from(&mut self, src: &[T]) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        unsafe { contexted_call!(self, cuMemcpy3D_v2, &memcpy3d_param_h2a(src, self)) }
            .expect("memcpy into array failed");
    }

    fn copy_from_async<'a>(&'a mut self, src: &'a [T]) -> BoxFuture<'a, ()> {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        let stream = stream::Stream::new(self.context.get_ref());
        unsafe {
            contexted_call!(
                self,
                cuMemcpy3DAsync_v2,
                &memcpy3d_param_h2a(src, self),
                stream.stream
            )
        }
        .expect("memcpy into array failed");
        Box::pin(async { stream.into_future().await.expect("async memcpy failed") })
    }
}

fn memcpy3d_param_a2h<T: Scalar, Dim: Dimension>(
    src: &Array<T, Dim>,
    dst: &mut [T],
) -> CUDA_MEMCPY3D {
    let dim = src.dim;
    CUDA_MEMCPY3D {
        srcMemoryType: CUmemorytype_enum::CU_MEMORYTYPE_ARRAY,
        srcArray: src.array,

        dstMemoryType: CUmemorytype_enum::CU_MEMORYTYPE_UNIFIED,
        dstDevice: dst.as_mut_ptr() as CUdeviceptr,

        WidthInBytes: dim.width() * T::size_of() * dim.num_channels().to_usize().unwrap(),
        Height: dim.height(),
        Depth: dim.depth(),

        ..Default::default()
    }
}

impl<T: Scalar, Dim: Dimension> Memcpy<Array<T, Dim>> for [T] {
    fn copy_from(&mut self, src: &Array<T, Dim>) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        unsafe { contexted_call!(src, cuMemcpy3D_v2, &memcpy3d_param_a2h(src, self)) }
            .expect("memcpy from array failed");
    }

    fn copy_from_async<'a>(&'a mut self, src: &'a Array<T, Dim>) -> BoxFuture<'a, ()> {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        let stream = stream::Stream::new(src.context.get_ref());
        unsafe {
            contexted_call!(
                src,
                cuMemcpy3DAsync_v2,
                &memcpy3d_param_a2h(src, self),
                stream.stream
            )
        }
        .expect("memcpy from array failed");
        Box::pin(async { stream.into_future().await.expect("async memcpy failed") })
    }
}

macro_rules! impl_memcpy_array {
    ($t:path) => {
        impl<T: Scalar, Dim: Dimension> Memcpy<Array<T, Dim>> for $t {
            fn copy_from(&mut self, src: &Array<T, Dim>) {
                self.as_mut_slice().copy_from(src);
            }
            fn copy_from_async<'a>(&'a mut self, src: &'a Array<T, Dim>) -> BoxFuture<'a, ()> {
                self.as_mut_slice().copy_from_async(src)
            }
        }

        impl<T: Scalar, Dim: Dimension> Memcpy<$t> for Array<T, Dim> {
            fn copy_from(&mut self, src: &$t) {
                self.copy_from(src.as_slice());
            }
            fn copy_from_async<'a>(&'a mut self, src: &'a $t) -> BoxFuture<'a, ()> {
                self.copy_from_async(src.as_slice())
            }
        }
    };
}

impl_memcpy_array!(DeviceMemory::<T>);
impl_memcpy_array!(PageLockedMemory::<T>);
impl_memcpy_array!(RegisteredMemory::<'_, T>);

impl<T: Scalar, Dim: Dimension> Memset for Array<T, Dim> {
    fn set(&mut self, value: Self::Elem) {
        // FIXME CUDA does not have memcpy for array. This is easy but too expensive alternative way
        let src = PageLockedMemory::from_elem(&self.context, self.dim.len(), value);
        self.copy_from(&src);
    }
}

impl<T: Scalar, Dim: Dimension> Allocatable for Array<T, Dim> {
    type Shape = Dim;
    unsafe fn uninitialized(context: &Context, dim: Dim) -> Self {
        let desc = dim.as_descriptor::<T>();
        let array =
            contexted_new!(context, cuArray3DCreate_v2, &desc).expect("Cannot create a new array");
        Array {
            array,
            dim,
            context: context.clone(),
            phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::*;

    #[test]
    fn new_1d() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let _array1: Array<f32, Ix1> = Array::zeros(&context, 10.into());
        let _array2: Array<f32, Ix1> = Array::zeros(&context, (10,).into());
        Ok(())
    }

    #[test]
    fn new_2d() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let _array: Array<f32, Ix2> = Array::zeros(&context, (10, 12).into());
        Ok(())
    }

    #[test]
    fn new_3d() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let _array: Array<f32, Ix3> = Array::zeros(&context, (10, 12, 8).into());
        Ok(())
    }

    #[test]
    fn new_1d_layered() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let _array: Array<f32, Ix1Layered> = Array::zeros(&context, (10, 12).into());
        Ok(())
    }

    #[test]
    fn new_2d_layered() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let _array: Array<f32, Ix2Layered> = Array::zeros(&context, (10, 12, 8).into());
        Ok(())
    }

    #[test]
    fn memcpy_h2a2h_1d() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let n = 10;
        let src = PageLockedMemory::from_elem(&context, n, 2_u32);
        let mut dst = PageLockedMemory::zeros(&context, n);
        let mut array = unsafe { Array::<u32, Ix1>::uninitialized(&context, n.into()) };
        array.copy_from(&src);
        dst.copy_from(&array);
        dbg!(dst.as_slice());
        for i in 0..n {
            assert_eq!(dst[i], 2_u32);
        }
        Ok(())
    }

    #[test]
    fn memcpy_d2a2d_2d() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let n = 3;
        let m = 4;
        let src = DeviceMemory::from_elem(&context, n * m, 2_u32);
        let mut dst = DeviceMemory::zeros(&context, n * m);
        let mut array = unsafe { Array::<u32, Ix2>::uninitialized(&context, (n, m).into()) };
        array.copy_from(&src);
        dst.copy_from(&array);
        dbg!(dst.as_slice());
        for i in 0..n * m {
            assert_eq!(dst[i], 2_u32);
        }
        Ok(())
    }

    #[test]
    fn memcpy_h2a2h_2d() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let n = 3;
        let m = 4;
        let src = PageLockedMemory::from_elem(&context, n * m, 2_u32);
        let mut dst = PageLockedMemory::zeros(&context, n * m);
        let mut array = unsafe { Array::<u32, Ix2>::uninitialized(&context, (n, m).into()) };
        array.copy_from(&src);
        dst.copy_from(&array);
        dbg!(dst.as_slice());
        for i in 0..n {
            assert_eq!(dst[i], 2_u32);
        }
        Ok(())
    }

    #[test]
    fn memcpy_d2a2d_1d() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let n = 3;
        let m = 4;
        let src = DeviceMemory::from_elem(&context, n * m, 2_u32);
        let mut dst = DeviceMemory::zeros(&context, n * m);
        let mut array = unsafe { Array::<u32, Ix2>::uninitialized(&context, (n, m).into()) };
        array.copy_from(&src);
        dst.copy_from(&array);
        dbg!(dst.as_slice());
        for i in 0..n {
            assert_eq!(dst[i], 2_u32);
        }
        Ok(())
    }
    #[test]
    fn memcpy_h2a2h_3d() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let n = 3;
        let m = 4;
        let l = 2;
        let src = PageLockedMemory::from_elem(&context, n * m * l, 2_u32);
        let mut dst = PageLockedMemory::zeros(&context, n * m * l);
        let mut array = unsafe { Array::<u32, Ix3>::uninitialized(&context, (n, m, l).into()) };
        array.copy_from(&src);
        dst.copy_from(&array);
        dbg!(dst.as_slice());
        for i in 0..n {
            assert_eq!(dst[i], 2_u32);
        }
        Ok(())
    }

    #[test]
    fn memcpy_d2a2d_3d() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let n = 3;
        let m = 4;
        let l = 2;
        let src = DeviceMemory::from_elem(&context, n * l * m, 2_u32);
        let mut dst = DeviceMemory::zeros(&context, n * l * m);
        let mut array = unsafe { Array::<u32, Ix3>::uninitialized(&context, (n, m, l).into()) };
        array.copy_from(&src);
        dst.copy_from(&array);
        dbg!(dst.as_slice());
        for i in 0..n {
            assert_eq!(dst[i], 2_u32);
        }
        Ok(())
    }

    #[test]
    fn memcpy_h2a2h_1dlayer() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let n = 3;
        let m = 4;
        let src = PageLockedMemory::from_elem(&context, n * m, 2_u32);
        let mut dst = PageLockedMemory::zeros(&context, n * m);
        let mut array = unsafe { Array::<u32, Ix1Layered>::uninitialized(&context, (n, m).into()) };
        array.copy_from(&src);
        dst.copy_from(&array);
        dbg!(dst.as_slice());
        for i in 0..n {
            assert_eq!(dst[i], 2_u32);
        }
        Ok(())
    }

    #[test]
    fn memcpy_d2a2d_1dlayer() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let n = 3;
        let m = 4;
        let src = DeviceMemory::from_elem(&context, n * m, 2_u32);
        let mut dst = DeviceMemory::zeros(&context, n * m);
        let mut array = unsafe { Array::<u32, Ix1Layered>::uninitialized(&context, (n, m).into()) };
        array.copy_from(&src);
        dst.copy_from(&array);
        dbg!(dst.as_slice());
        for i in 0..n {
            assert_eq!(dst[i], 2_u32);
        }
        Ok(())
    }

    #[test]
    fn memcpy_h2a2h_2dlayer() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let n = 3;
        let m = 4;
        let l = 2;
        let src = PageLockedMemory::from_elem(&context, n * m * l, 2_u32);
        let mut dst = PageLockedMemory::zeros(&context, n * m * l);
        let mut array =
            unsafe { Array::<u32, Ix2Layered>::uninitialized(&context, (n, m, l).into()) };
        array.copy_from(&src);
        dst.copy_from(&array);
        dbg!(dst.as_slice());
        for i in 0..n {
            assert_eq!(dst[i], 2_u32);
        }
        Ok(())
    }

    #[test]
    fn memcpy_d2a2d_2dlayer() -> Result<()> {
        let device = Device::nth(0)?;
        let context = device.create_context();
        let n = 3;
        let m = 4;
        let l = 2;
        let src = DeviceMemory::from_elem(&context, n * m * l, 2_u32);
        let mut dst = DeviceMemory::zeros(&context, n * m * l);
        let mut array =
            unsafe { Array::<u32, Ix2Layered>::uninitialized(&context, (n, m, l).into()) };
        array.copy_from(&src);
        dst.copy_from(&array);
        dbg!(dst.as_slice());
        for i in 0..n {
            assert_eq!(dst[i], 2_u32);
        }
        Ok(())
    }
}
