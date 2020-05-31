use num_traits::ToPrimitive;

/// Size of Block (thread block) in [CUDA thread hierarchy]( http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model )
///
/// Every input integer and float convert into `u32` using [ToPrimitive].
/// If the conversion is impossible, e.g. negative or too large integers, the conversion will panics.
///
/// [ToPrimitive]: https://docs.rs/num-traits/0.2.11/num_traits/cast/trait.ToPrimitive.html
///
/// Examples
/// --------
///
/// - Explicit creation
///
/// ```
/// # use accel::*;
/// let block1d = Block::x(64);
/// assert_eq!(block1d.x, 64);
///
/// let block2d = Block::xy(64, 128);
/// assert_eq!(block2d.x, 64);
/// assert_eq!(block2d.y, 128);
///
/// let block3d = Block::xyz(64, 128, 256);
/// assert_eq!(block3d.x, 64);
/// assert_eq!(block3d.y, 128);
/// assert_eq!(block3d.z, 256);
/// ```
///
/// - From single integer (unsigned and signed)
///
/// ```
/// # use accel::*;
/// let block1d: Block = 64_usize.into();
/// assert_eq!(block1d.x, 64);
///
/// let block1d: Block = 64_i32.into();
/// assert_eq!(block1d.x, 64);
/// ```
///
/// - From tuple
///
/// ```
/// # use accel::*;
/// let block1d: Block = (64,).into();
/// assert_eq!(block1d.x, 64);
///
/// let block2d: Block = (64, 128).into();
/// assert_eq!(block2d.x, 64);
/// assert_eq!(block2d.y, 128);
///
/// let block3d: Block = (64, 128, 256).into();
/// assert_eq!(block3d.x, 64);
/// assert_eq!(block3d.y, 128);
/// assert_eq!(block3d.z, 256);
/// ```
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
