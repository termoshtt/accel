pub use cuda::CUarray_format as ArrayFormatTag;
use num_traits::Num;

pub trait Scalar: Num + Copy {
    fn format() -> ArrayFormatTag;

    fn size_of() -> usize {
        std::mem::size_of::<Self>()
    }

    /// Get little endian format in u8
    fn to_le_u8(self) -> Option<u8> {
        None
    }

    /// Get little endian format in u16
    fn to_le_u16(self) -> Option<u16> {
        None
    }

    /// Get little endian format in u32
    fn to_le_u32(self) -> Option<u32> {
        None
    }
}

macro_rules! impl_array_scalar {
    ($scalar:ty, $le:ty, $format:ident) => {
        impl Scalar for $scalar {
            fn format() -> ArrayFormatTag {
                ArrayFormatTag::$format
            }
            paste::item! {
                fn [< to_le_ $le >](self) -> Option<$le> {
                    Some($le::from_le_bytes(self.to_le_bytes()))
                }
            }
        }
    };
}

impl_array_scalar!(u8, u8, CU_AD_FORMAT_UNSIGNED_INT8);
impl_array_scalar!(u16, u16, CU_AD_FORMAT_UNSIGNED_INT16);
impl_array_scalar!(u32, u32, CU_AD_FORMAT_UNSIGNED_INT32);
impl_array_scalar!(i8, u8, CU_AD_FORMAT_SIGNED_INT8);
impl_array_scalar!(i16, u16, CU_AD_FORMAT_SIGNED_INT16);
impl_array_scalar!(i32, u32, CU_AD_FORMAT_SIGNED_INT32);
// FIXME f16 is not supported yet
// impl_array_scalar!(f16, u16, CU_AD_FORMAT_HALF);
impl_array_scalar!(f32, u32, CU_AD_FORMAT_FLOAT);

impl Scalar for f64 {
    fn format() -> ArrayFormatTag {
        panic!("CUDA Array does not support f64");
    }
}
