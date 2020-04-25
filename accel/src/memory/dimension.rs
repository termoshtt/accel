use crate::*;
use derive_new::new;
use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::{ToPrimitive, Zero};
use std::{fmt::Debug, ops::Add};

pub use cuda::CUDA_ARRAY3D_DESCRIPTOR as Descriptor;

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

pub trait Dimension: Zero + Debug + Clone + Copy + PartialEq {
    fn as_descriptor<T: Scalar>(&self) -> Descriptor;

    /// Number of elements
    fn len(&self) -> usize;

    /// Get number of element `T` in each "CUDA Array element"
    fn num_channels(&self) -> NumChannels;

    fn width(&self) -> usize {
        self.as_descriptor::<u32>().Width
    }

    fn height(&self) -> usize {
        std::cmp::max(self.as_descriptor::<u32>().Height, 1)
    }

    fn depth(&self) -> usize {
        std::cmp::max(self.as_descriptor::<u32>().Depth, 1)
    }
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
    pub height: usize,
    #[new(default)]
    pub num_channels: NumChannels,
}

impl From<(usize, usize)> for Ix2 {
    fn from((width, height): (usize, usize)) -> Ix2 {
        Ix2 {
            width,
            height,
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
            height: self.height + other.height,
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
            Height: self.height,
            Depth: 0,
            NumChannels: self.num_channels.to_u32().unwrap(),
            Flags: ArrayFlag::empty().bits(),
            Format: T::format(),
        }
    }

    fn len(&self) -> usize {
        self.width * self.height * self.num_channels.to_usize().unwrap()
    }

    fn num_channels(&self) -> NumChannels {
        self.num_channels
    }
}

/// Spec of 3D Array
#[derive(Debug, Clone, Copy, PartialEq, new)]
pub struct Ix3 {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    #[new(default)]
    pub num_channels: NumChannels,
}

impl From<(usize, usize, usize)> for Ix3 {
    fn from((width, height, depth): (usize, usize, usize)) -> Ix3 {
        Ix3 {
            width,
            height,
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
            height: self.height + other.height,
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
            Height: self.height,
            Depth: self.depth,
            NumChannels: self.num_channels.to_u32().unwrap(),
            Flags: ArrayFlag::empty().bits(),
            Format: T::format(),
        }
    }

    fn len(&self) -> usize {
        self.width * self.height * self.depth * self.num_channels().to_usize().unwrap()
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
    /// height of each layer
    pub height: usize,
    /// Depth of layer
    pub depth: usize,
    #[new(default)]
    pub num_channels: NumChannels,
}

impl From<(usize, usize, usize)> for Ix2Layered {
    fn from((width, height, depth): (usize, usize, usize)) -> Ix2Layered {
        Ix2Layered {
            width,
            height,
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
            height: self.height + other.height,
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
            Height: self.height,
            Depth: self.depth,
            NumChannels: self.num_channels.to_u32().unwrap(),
            Flags: ArrayFlag::LAYERED.bits(),
            Format: T::format(),
        }
    }

    fn len(&self) -> usize {
        self.width * self.height * self.depth * self.num_channels.to_usize().unwrap()
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
