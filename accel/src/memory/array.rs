//! CUDA [Array] and [Texture], [Surface] Objects
//!
//! [Array]:   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-arrays
//! [Texture]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TEXOBJECT.html#group__CUDA__TEXOBJECT
//! [Surface]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__SURFOBJECT.html#group__CUDA__SURFOBJECT

use crate::{contexted_call, contexted_new, device::Contexted, *};
use cuda::*;
use derive_new::new;
use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::{ToPrimitive, Zero};
use std::{marker::PhantomData, ops::Add};

pub use cuda::CUDA_ARRAY3D_DESCRIPTOR as Descriptor;

#[derive(Debug)]
pub struct Array<'ctx, T, Dim> {
    array: CUarray,
    dim: Dim,
    ctx: &'ctx Context,
    phantom: PhantomData<T>,
}

impl<T, Dim> Drop for Array<'_, T, Dim> {
    fn drop(&mut self) {
        if let Err(e) = unsafe { contexted_call!(self, cuArrayDestroy, self.array) } {
            log::error!("Failed to cleanup array: {:?}", e);
        }
    }
}

impl<'ctx, T: Scalar, Dim: Dimension> Array<'ctx, T, Dim> {
    /// Get dimension
    pub fn dim(&self) -> &Dim {
        &self.dim
    }
}

impl<'ctx, T: Scalar, Dim: Dimension> Memory for Array<'ctx, T, Dim> {
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

    fn try_get_context(&self) -> Option<&Context> {
        Some(self.get_context())
    }
}

impl<T: Scalar> Memcpy<PageLockedMemory<'_, T>> for Array<'_, T, Ix1> {
    fn copy_from(&mut self, src: &PageLockedMemory<'_, T>) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        unsafe {
            contexted_call!(
                self.get_context(),
                cuMemcpyHtoA_v2,
                self.array,
                0, /* offset */
                src.head_addr() as *const _,
                src.num_elem() * T::size_of()
            )
        }
        .expect("memcpy from Array to page-locked host memory failed");
    }

    fn copy_to(&self, dest: &mut PageLockedMemory<'_, T>) {
        assert_ne!(self.head_addr(), dest.head_addr());
        assert_eq!(self.num_elem(), dest.num_elem());
        unsafe {
            contexted_call!(
                self.get_context(),
                cuMemcpyAtoH_v2,
                dest.head_addr_mut() as *mut _,
                self.array,
                0, /* offset */
                dest.num_elem() * T::size_of()
            )
        }
        .expect("memcpy from Array to page-locked host memory failed");
    }
}

// use default impl
impl<T: Scalar> Memcpy<Array<'_, T, Ix1>> for PageLockedMemory<'_, T> {}

impl<T: Scalar> Memcpy<DeviceMemory<'_, T>> for Array<'_, T, Ix1> {
    fn copy_from(&mut self, src: &DeviceMemory<'_, T>) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        unsafe {
            contexted_call!(
                self.get_context(),
                cuMemcpyDtoA_v2,
                self.array,
                0, /* offset */
                src.head_addr() as CUdeviceptr,
                src.num_elem() * T::size_of()
            )
        }
        .expect("memcpy from Device to Array failed");
    }

    fn copy_to(&self, dest: &mut DeviceMemory<'_, T>) {
        assert_ne!(self.head_addr(), dest.head_addr());
        assert_eq!(self.num_elem(), dest.num_elem());
        unsafe {
            contexted_call!(
                self.get_context(),
                cuMemcpyAtoD_v2,
                dest.head_addr_mut() as CUdeviceptr,
                self.array,
                0, /* offset */
                dest.num_elem() * T::size_of()
            )
        }
        .expect("memcpy from Array to Device failed");
    }
}

// use default impl
impl<T: Scalar> Memcpy<Array<'_, T, Ix1>> for DeviceMemory<'_, T> {}

impl<'ctx, T: Scalar, Dim: Dimension> Memset for Array<'ctx, T, Dim> {
    fn set(&mut self, _value: Self::Elem) {
        todo!()
    }
}

impl<T, Dim> Contexted for Array<'_, T, Dim> {
    fn get_context(&self) -> &Context {
        self.ctx
    }
}

impl<'ctx, T: Scalar, Dim: Dimension> Allocatable<'ctx> for Array<'ctx, T, Dim> {
    type Shape = Dim;
    unsafe fn uninitialized(ctx: &'ctx Context, dim: Dim) -> Self {
        let _gurad = ctx.guard_context();
        let desc = dim.as_descriptor::<T>();
        let array =
            contexted_new!(ctx, cuArray3DCreate_v2, &desc).expect("Cannot create a new array");
        Array {
            array,
            dim,
            ctx,
            phantom: PhantomData,
        }
    }
}

/// This specifies the number of packed elements per "CUDA array element".
///
/// - The CUDA array element approach is useful e.g. for [RGBA color model],
///   which has 4 values at each point of figures.
/// - For example, When `T=f32` and `NumChannels::Two`,
///   the size of "CUDA array element" is 64bit as packed two 32bit float values.
/// - We call `T` element, although "CUDA array element" represents `[T; num_channels]`.
///   `Memory::num_elem()` returns how many `T` exists in this array.
///
/// [RGBA color model]: https://en.wikipedia.org/wiki/RGBA_color_model
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, FromPrimitive, ToPrimitive)]
pub enum NumChannels {
    /// Single element in each "CUDA Array element"
    One = 1,
    /// Two scalars in each CUDA Array element
    Two = 2,
    /// Four scalars in each CUDA Array element
    Four = 4,
}

impl Default for NumChannels {
    fn default() -> Self {
        NumChannels::One
    }
}

pub trait Dimension: Zero {
    fn as_descriptor<T: Scalar>(&self) -> Descriptor;

    /// Number of elements
    fn len(&self) -> usize;

    /// Get number of element `T` in each "CUDA Array element"
    fn num_channels(&self) -> NumChannels;
}

/// Spec of 1D Array
#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix1 {
    pub width: usize,
    #[new(default)]
    pub num_channels: NumChannels,
}

impl From<usize> for Ix1 {
    fn from(width: usize) -> Ix1 {
        Ix1 {
            width,
            num_channels: NumChannels::One,
        }
    }
}

impl From<(usize,)> for Ix1 {
    fn from((width,): (usize,)) -> Ix1 {
        Ix1 {
            width,
            num_channels: NumChannels::One,
        }
    }
}

impl Add for Ix1 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        assert_eq!(self.num_channels, other.num_channels);
        Self {
            width: self.width + other.width,
            num_channels: self.num_channels,
        }
    }
}

impl Zero for Ix1 {
    fn zero() -> Self {
        Ix1::new(0)
    }

    fn is_zero(&self) -> bool {
        self.len() == 0
    }
}

impl Dimension for Ix1 {
    fn as_descriptor<T: Scalar>(&self) -> Descriptor {
        Descriptor {
            Width: self.width,
            Height: 0,
            Depth: 0,
            NumChannels: self.num_channels.to_u32().unwrap(),
            Flags: ArrayFlag::empty().bits(),
            Format: T::format(),
        }
    }

    fn len(&self) -> usize {
        self.width * self.num_channels.to_usize().unwrap()
    }

    fn num_channels(&self) -> NumChannels {
        self.num_channels
    }
}

/// Spec of 2D Array
#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix2 {
    pub width: usize,
    pub hight: usize,
    #[new(default)]
    pub num_channels: NumChannels,
}

impl From<(usize, usize)> for Ix2 {
    fn from((width, hight): (usize, usize)) -> Ix2 {
        Ix2 {
            width,
            hight,
            num_channels: NumChannels::One,
        }
    }
}

impl Add for Ix2 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        assert_eq!(self.num_channels, other.num_channels);
        Self {
            width: self.width + other.width,
            hight: self.hight + other.hight,
            num_channels: self.num_channels,
        }
    }
}

impl Zero for Ix2 {
    fn zero() -> Self {
        Ix2::new(0, 0)
    }

    fn is_zero(&self) -> bool {
        self.len() == 0
    }
}

impl Dimension for Ix2 {
    fn as_descriptor<T: Scalar>(&self) -> Descriptor {
        Descriptor {
            Width: self.width,
            Height: self.hight,
            Depth: 0,
            NumChannels: self.num_channels.to_u32().unwrap(),
            Flags: ArrayFlag::empty().bits(),
            Format: T::format(),
        }
    }

    fn len(&self) -> usize {
        self.width * self.hight * self.num_channels.to_usize().unwrap()
    }

    fn num_channels(&self) -> NumChannels {
        self.num_channels
    }
}

/// Spec of 3D Array
#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix3 {
    pub width: usize,
    pub hight: usize,
    pub depth: usize,
    #[new(default)]
    pub num_channels: NumChannels,
}

impl From<(usize, usize, usize)> for Ix3 {
    fn from((width, hight, depth): (usize, usize, usize)) -> Ix3 {
        Ix3 {
            width,
            hight,
            depth,
            num_channels: NumChannels::One,
        }
    }
}

impl Add for Ix3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        assert_eq!(self.num_channels, other.num_channels);
        Self {
            width: self.width + other.width,
            hight: self.hight + other.hight,
            depth: self.depth + other.depth,
            num_channels: self.num_channels,
        }
    }
}

impl Zero for Ix3 {
    fn zero() -> Self {
        Ix3::new(0, 0, 0)
    }

    fn is_zero(&self) -> bool {
        self.len() == 0
    }
}

impl Dimension for Ix3 {
    fn as_descriptor<T: Scalar>(&self) -> Descriptor {
        Descriptor {
            Width: self.width,
            Height: self.hight,
            Depth: self.depth,
            NumChannels: self.num_channels.to_u32().unwrap(),
            Flags: ArrayFlag::empty().bits(),
            Format: T::format(),
        }
    }

    fn len(&self) -> usize {
        self.width * self.hight * self.depth * self.num_channels().to_usize().unwrap()
    }

    fn num_channels(&self) -> NumChannels {
        self.num_channels
    }
}

/// Spec of Layered 1D Array
#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix1Layered {
    /// Width of each layer
    pub width: usize,
    /// Depth of layer
    pub depth: usize,
    #[new(default)]
    pub num_channels: NumChannels,
}

impl From<(usize, usize)> for Ix1Layered {
    fn from((width, depth): (usize, usize)) -> Ix1Layered {
        Ix1Layered {
            width,
            depth,
            num_channels: NumChannels::One,
        }
    }
}

impl Add for Ix1Layered {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        assert_eq!(self.num_channels, other.num_channels);
        Self {
            width: self.width + other.width,
            depth: self.depth + other.depth,
            num_channels: self.num_channels,
        }
    }
}

impl Zero for Ix1Layered {
    fn zero() -> Self {
        Self::new(0, 0)
    }

    fn is_zero(&self) -> bool {
        self.len() == 0
    }
}

impl Dimension for Ix1Layered {
    fn as_descriptor<T: Scalar>(&self) -> Descriptor {
        Descriptor {
            Width: self.width,
            Height: 0,
            Depth: self.depth,
            NumChannels: self.num_channels.to_u32().unwrap(),
            Flags: ArrayFlag::LAYERED.bits(),
            Format: T::format(),
        }
    }

    fn len(&self) -> usize {
        self.width * self.depth * self.num_channels.to_usize().unwrap()
    }

    fn num_channels(&self) -> NumChannels {
        self.num_channels
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
    #[new(default)]
    pub num_channels: NumChannels,
}

impl From<(usize, usize, usize)> for Ix2Layered {
    fn from((width, hight, depth): (usize, usize, usize)) -> Ix2Layered {
        Ix2Layered {
            width,
            hight,
            depth,
            num_channels: NumChannels::One,
        }
    }
}

impl Add for Ix2Layered {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        assert_eq!(self.num_channels, other.num_channels);
        Self {
            width: self.width + other.width,
            hight: self.hight + other.hight,
            depth: self.depth + other.depth,
            num_channels: self.num_channels,
        }
    }
}

impl Zero for Ix2Layered {
    fn zero() -> Self {
        Self::new(0, 0, 0)
    }

    fn is_zero(&self) -> bool {
        self.len() == 0
    }
}

impl Dimension for Ix2Layered {
    fn as_descriptor<T: Scalar>(&self) -> Descriptor {
        Descriptor {
            Width: self.width,
            Height: self.hight,
            Depth: self.depth,
            NumChannels: self.num_channels.to_u32().unwrap(),
            Flags: ArrayFlag::LAYERED.bits(),
            Format: T::format(),
        }
    }

    fn len(&self) -> usize {
        self.width * self.hight * self.depth * self.num_channels.to_usize().unwrap()
    }

    fn num_channels(&self) -> NumChannels {
        self.num_channels
    }
}

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
    use super::*;
    use crate::{device::*, error::*};

    #[test]
    fn new_1d() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _array1: Array<f32, Ix1> = Array::zeros(&ctx, 10.into());
        let _array2: Array<f32, Ix1> = Array::zeros(&ctx, (10,).into());
        Ok(())
    }

    #[test]
    fn new_2d() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _array: Array<f32, Ix2> = Array::zeros(&ctx, (10, 12).into());
        Ok(())
    }

    #[test]
    fn new_3d() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _array: Array<f32, Ix3> = Array::zeros(&ctx, (10, 12, 8).into());
        Ok(())
    }

    #[test]
    fn new_1d_layered() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _array: Array<f32, Ix1Layered> = Array::zeros(&ctx, (10, 12).into());
        Ok(())
    }

    #[test]
    fn new_2d_layered() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _array: Array<f32, Ix2Layered> = Array::zeros(&ctx, (10, 12, 8).into());
        Ok(())
    }
}
