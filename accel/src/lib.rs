//! GPGPU framework for Rust

extern crate cuda_driver_sys as cuda;
extern crate cuda_runtime_sys as cudart;

pub mod device;
pub mod driver;
pub mod error;
pub mod mvec;
pub mod uvec;

pub use driver::module::{Block, Grid};
pub use mvec::MVec;
pub use uvec::UVec;

#[macro_export]
macro_rules! ffi_call {
    ($ffi:path, $($args:expr),*) => {
        $ffi($($args),*).check(stringify!($ffi))
    };
    ($ffi:path) => {
        $ffi().check(stringify!($ffi))
    };
}

#[macro_export]
macro_rules! ffi_call_unsafe {
    ($ffi:path, $($args:expr),*) => {
        unsafe { $crate::error::Check::check($ffi($($args),*), stringify!($ffi)) }
    };
    ($ffi:path) => {
        unsafe { $crate::error::Check::check($ffi(), stringify!($ffi)) }
    };
}

#[macro_export]
macro_rules! ffi_new {
    ($ffi:path, $($args:expr),*) => {
        {
            let mut value = ::std::mem::MaybeUninit::uninit();
            $crate::error::Check::check($ffi(value.as_mut_ptr(), $($args),*), stringify!($ffi)).map(|_| value.assume_init())
        }
    };
    ($ffi:path) => {
        {
            let mut value = ::std::mem::MaybeUninit::uninit();
            $crate::error::Check::check($ffi(value.as_mut_ptr()), stringify!($ffi)).map(|_| value.assume_init())
        }
    };
}

#[macro_export]
macro_rules! ffi_new_unsafe {
    ($ffi:path, $($args:expr),*) => {
        unsafe { $crate::ffi_new!($ffi, $($args),*) }
    };
    ($ffi:path) => {
        unsafe { $crate::ffi_new!($ffi) }
    };
}
