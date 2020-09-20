//! Support crate for writting GPU kernel in Rust
//!
//! - This crate works only for `nvptx64-nvidia-cuda` target
//! - There is no support of `libstd` for `nvptx64-nvidia-cuda` target,
//!   i.e. You need to write `#![no_std]` Rust code.
//! - `alloc` crate is supported by `accel_core::PTXAllocator` which utilizes CUDA malloc/free system-calls
//!   - You can use `println!` and `assert_eq!` throught it.

#![feature(stdsimd)]
#![no_std]

extern crate alloc;

use alloc::alloc::*;
use core::arch::nvptx;

/// Memory allocator using CUDA malloc/free
pub struct PTXAllocator;

unsafe impl GlobalAlloc for PTXAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        nvptx::malloc(layout.size()) as *mut u8
    }
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        nvptx::free(ptr as *mut _);
    }
}

/// Alternative of [std::print!](https://doc.rust-lang.org/std/macro.print.html) using CUDA `vprintf` system-call
#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {
        let msg = ::alloc::format!($($arg)*);
        unsafe {
            ::core::arch::nvptx::vprintf(msg.as_ptr(), ::core::ptr::null_mut());
        }
    }
}

/// Alternative of [std::println!](https://doc.rust-lang.org/std/macro.println.html) using CUDA `vprintf` system-call
#[macro_export]
macro_rules! println {
    () => ($crate::print!("\n"));
    ($fmt:expr) => ($crate::print!(concat!($fmt, "\n")));
    ($fmt:expr, $($arg:tt)*) => ($crate::print!(concat!($fmt, "\n"), $($arg)*));
}

/// Assertion in GPU kernel for two expressions are equal.
///
/// If assertion failed, accel API will return [accel::error::AccelError::DeviceAssertionFailed](https://docs.rs/accel/0.3.0-alpha.2/accel/error/enum.AccelError.html#variant.DeviceAssertionFailed)
#[macro_export]
macro_rules! assert_eq {
    ($a:expr, $b:expr) => {
        if $a != $b {
            let msg = alloc::format!(
                "\nassertion failed: ({} == {})\nleft : {:?}\nright: {:?}",
                stringify!($a),
                stringify!($b),
                $a,
                $b
            );
            unsafe {
                ::core::arch::nvptx::__assert_fail(
                    msg.as_ptr(),
                    file!().as_ptr(),
                    line!(),
                    // FIXME cannot get function name.
                    // See https://github.com/rust-lang/rfcs/pull/2818
                    "".as_ptr(),
                )
            };
        }
    };
}

/// Assertion in GPU kernel for two expressions are not equal.
///
/// If assertion failed, accel API will return [accel::error::AccelError::DeviceAssertionFailed](https://docs.rs/accel/0.3.0-alpha.2/accel/error/enum.AccelError.html#variant.DeviceAssertionFailed)
#[macro_export]
macro_rules! assert_ne {
    ($a:expr, $b:expr) => {
        if $a == $b {
            let msg = alloc::format!(
                "\nassertion failed: ({} != {})\nleft : {:?}\nright: {:?}",
                stringify!($a),
                stringify!($b),
                $a,
                $b
            );
            unsafe {
                ::core::arch::nvptx::__assert_fail(
                    msg.as_ptr(),
                    file!().as_ptr(),
                    line!(),
                    // FIXME cannot get function name.
                    // See https://github.com/rust-lang/rfcs/pull/2818
                    "".as_ptr(),
                )
            };
        }
    };
}

/// Dimension specified in kernel launching
pub struct Dim3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// Indices where the kernel code running on
pub struct Idx3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

pub fn block_dim() -> Dim3 {
    unsafe {
        Dim3 {
            x: nvptx::_block_dim_x(),
            y: nvptx::_block_dim_y(),
            z: nvptx::_block_dim_z(),
        }
    }
}

pub fn block_idx() -> Idx3 {
    unsafe {
        Idx3 {
            x: nvptx::_block_idx_x(),
            y: nvptx::_block_idx_y(),
            z: nvptx::_block_idx_z(),
        }
    }
}

pub fn grid_dim() -> Dim3 {
    unsafe {
        Dim3 {
            x: nvptx::_grid_dim_x(),
            y: nvptx::_grid_dim_y(),
            z: nvptx::_grid_dim_z(),
        }
    }
}

pub fn thread_idx() -> Idx3 {
    unsafe {
        Idx3 {
            x: nvptx::_thread_idx_x(),
            y: nvptx::_thread_idx_y(),
            z: nvptx::_thread_idx_z(),
        }
    }
}

impl Dim3 {
    pub fn size(&self) -> i32 {
        (self.x * self.y * self.z)
    }
}

impl Idx3 {
    pub fn into_id(&self, dim: Dim3) -> i32 {
        self.x + self.y * dim.x + self.z * dim.x * dim.y
    }
}

pub fn index() -> isize {
    let block_id = block_idx().into_id(grid_dim());
    let thread_id = thread_idx().into_id(block_dim());
    (block_id * block_dim.size() + thread_id) as isize
}
