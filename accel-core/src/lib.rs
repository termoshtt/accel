#![feature(stdsimd)]
#![no_std]

use core::arch::nvptx;

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
