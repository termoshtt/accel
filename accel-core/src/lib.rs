#![feature(stdsimd)]
#![no_std]

extern crate alloc;

use alloc::alloc::*;
use core::arch::nvptx;

pub struct PTXAllocator;

unsafe impl GlobalAlloc for PTXAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        nvptx::malloc(layout.size()) as *mut u8
    }
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        nvptx::free(ptr as *mut _);
    }
}

#[macro_export]
macro_rules! assert_eq {
    ($a:expr, $b:expr) => {
        if $a != $b {
            // FIXME show $a, $b, and their values
            let msg = "not equal";
            // FIXME cannot get function name.
            // See https://github.com/rust-lang/rfcs/pull/2818
            let func_name = "";
            unsafe {
                ::core::arch::nvptx::__assert_fail(
                    msg.as_ptr(),
                    file!().as_ptr(),
                    line!(),
                    func_name.as_ptr(),
                )
            };
        }
    };
}

#[macro_export]
macro_rules! assert_ne {
    ($a:expr, $b:expr) => {
        if $a == $b {
            // FIXME show $a, $b, and their values
            let msg = "not equal";
            // FIXME cannot get function name.
            // See https://github.com/rust-lang/rfcs/pull/2818
            let func_name = "";
            unsafe {
                ::core::arch::nvptx::__assert_fail(
                    msg.as_ptr(),
                    file!().as_ptr(),
                    line!(),
                    func_name.as_ptr(),
                )
            };
        }
    };
}

pub struct Dim3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

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
    (block_id + thread_id) as isize
}
