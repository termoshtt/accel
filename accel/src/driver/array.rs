use super::context::*;
use crate::{ffi_call_unsafe, ffi_new_unsafe};
use cuda::*;
use derive_new::new;
use std::marker::PhantomData;

pub use cuda::CUarray_format as ArrayFormatTag;
pub use cuda::CUDA_ARRAY3D_DESCRIPTOR as Descriptor;

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
    pub fn new(ctx: &Context, dim: impl Into<Dim>, num_channels: u32) -> Self {
        ctx.assure_current()
            .expect("Array::new requires valid and current context");
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
    fn as_descriptor<T: Scalar>(&self, num_channels: u32) -> Descriptor;
    /// Number of elements
    fn len(&self) -> usize;
}

/// Spec of 1D Array
#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix1 {
    pub width: usize,
}

impl Dimension for Ix1 {
    fn as_descriptor<T: Scalar>(&self, num_channels: u32) -> Descriptor {
        assert!(
            (num_channels == 1) || (num_channels == 2) || (num_channels == 4),
            "num_channels must be one of 1,2,4"
        );
        Descriptor {
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

/// Spec of 2D Array
#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix2 {
    pub width: usize,
    pub hight: usize,
}

impl Dimension for Ix2 {
    fn as_descriptor<T: Scalar>(&self, num_channels: u32) -> Descriptor {
        Descriptor {
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

/// Spec of 3D Array
#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix3 {
    pub width: usize,
    pub hight: usize,
    pub depth: usize,
}

impl Dimension for Ix3 {
    fn as_descriptor<T: Scalar>(&self, num_channels: u32) -> Descriptor {
        Descriptor {
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

/// Spec of Layered 1D Array
#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix1Layered {
    /// Width of each layer
    pub width: usize,
    /// Depth of layer
    pub depth: usize,
}

impl Dimension for Ix1Layered {
    fn as_descriptor<T: Scalar>(&self, num_channels: u32) -> Descriptor {
        Descriptor {
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

/// Spec of Layered 2D Array
#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix2Layered {
    /// Width of each layer
    pub width: usize,
    /// Hight of each layer
    pub hight: usize,
    /// Depth of layer
    pub depth: usize,
}

impl Dimension for Ix2Layered {
    fn as_descriptor<T: Scalar>(&self, num_channels: u32) -> Descriptor {
        Descriptor {
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
    fn format() -> ArrayFormatTag;
}

macro_rules! impl_array_scalar {
    ($scalar:ty, $format:ident) => {
        impl Scalar for $scalar {
            fn format() -> ArrayFormatTag {
                ArrayFormatTag::$format
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
    pub struct ArrayFlag: u32 {
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
    use crate::error::*;

    #[test]
    fn new_1d() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let _array: Array<f32, Ix1> = Array::new(&ctx, Ix1::new(10), 1);
        Ok(())
    }

    #[test]
    fn new_2d() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let _array: Array<f32, Ix2> = Array::new(&ctx, Ix2::new(10, 12), 1);
        Ok(())
    }

    #[test]
    fn new_3d() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let _array: Array<f32, Ix3> = Array::new(&ctx, Ix3::new(10, 12, 8), 1);
        Ok(())
    }

    #[test]
    fn new_1d_layered() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let _array: Array<f32, Ix1Layered> = Array::new(&ctx, Ix1Layered::new(10, 12), 1);
        Ok(())
    }

    #[test]
    fn new_2d_layered() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let _array: Array<f32, Ix2Layered> = Array::new(&ctx, Ix2Layered::new(10, 12, 8), 1);
        Ok(())
    }
}
