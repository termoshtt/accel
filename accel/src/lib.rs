//! GPGPU framework for Rust

extern crate cuda_driver_sys as cuda;

pub mod array;
pub mod device;
pub mod error;
pub mod instruction;
pub mod linker;
pub mod memory;
pub mod module;
pub mod stream;

pub use device::{Context, Device};
use num_traits::ToPrimitive;

/// Size of Block (thread block) in [CUDA thread hierarchy]( http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model )
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Block {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Block {
    /// 1D Block
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn x<I: ToPrimitive>(x: I) -> Self {
        Block {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: 1,
            z: 1,
        }
    }

    /// 2D Block
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn xy<I1: ToPrimitive, I2: ToPrimitive>(x: I1, y: I2) -> Self {
        Block {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: y.to_u32().expect("Cannot convert to u32"),
            z: 1,
        }
    }

    /// 3D Block
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn xyz<I1: ToPrimitive, I2: ToPrimitive, I3: ToPrimitive>(x: I1, y: I2, z: I3) -> Self {
        Block {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: y.to_u32().expect("Cannot convert to u32"),
            z: z.to_u32().expect("Cannot convert to u32"),
        }
    }
}

impl<I: ToPrimitive> Into<Block> for (I,) {
    fn into(self) -> Block {
        Block::x(self.0)
    }
}

impl<I1: ToPrimitive, I2: ToPrimitive> Into<Block> for (I1, I2) {
    fn into(self) -> Block {
        Block::xy(self.0, self.1)
    }
}

impl<I1: ToPrimitive, I2: ToPrimitive, I3: ToPrimitive> Into<Block> for (I1, I2, I3) {
    fn into(self) -> Block {
        Block::xyz(self.0, self.1, self.2)
    }
}

macro_rules! impl_into_block {
    ($integer:ty) => {
        impl Into<Block> for $integer {
            fn into(self) -> Block {
                Block::x(self)
            }
        }
    };
}

impl_into_block!(u8);
impl_into_block!(u16);
impl_into_block!(u32);
impl_into_block!(u64);
impl_into_block!(u128);
impl_into_block!(usize);
impl_into_block!(i8);
impl_into_block!(i16);
impl_into_block!(i32);
impl_into_block!(i64);
impl_into_block!(i128);
impl_into_block!(isize);

/// Size of Grid (grid of blocks) in [CUDA thread hierarchy]( http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model )
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Grid {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Grid {
    /// 1D Grid
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn x<I: ToPrimitive>(x: I) -> Self {
        Grid {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: 1,
            z: 1,
        }
    }

    /// 2D Grid
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn xy<I1: ToPrimitive, I2: ToPrimitive>(x: I1, y: I2) -> Self {
        Grid {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: y.to_u32().expect("Cannot convert to u32"),
            z: 1,
        }
    }

    /// 3D Grid
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn xyz<I1: ToPrimitive, I2: ToPrimitive, I3: ToPrimitive>(x: I1, y: I2, z: I3) -> Self {
        Grid {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: y.to_u32().expect("Cannot convert to u32"),
            z: z.to_u32().expect("Cannot convert to u32"),
        }
    }
}

impl<I: ToPrimitive> Into<Grid> for (I,) {
    fn into(self) -> Grid {
        Grid::x(self.0)
    }
}

impl<I1: ToPrimitive, I2: ToPrimitive> Into<Grid> for (I1, I2) {
    fn into(self) -> Grid {
        Grid::xy(self.0, self.1)
    }
}

impl<I1: ToPrimitive, I2: ToPrimitive, I3: ToPrimitive> Into<Grid> for (I1, I2, I3) {
    fn into(self) -> Grid {
        Grid::xyz(self.0, self.1, self.2)
    }
}
