#![feature(platform_intrinsics)]
#![no_std]

pub struct Dim3 {
    x: i32,
    y: i32,
    z: i32,
}

pub struct Idx3 {
    x: i32,
    y: i32,
    z: i32,
}

pub fn block_dim() -> Dim3 {
    unsafe {
        Dim3 {
            x: nvptx_block_dim_x(),
            y: nvptx_block_dim_y(),
            z: nvptx_block_dim_z(),
        }
    }
}

pub fn block_idx() -> Idx3 {
    unsafe {
        Idx3 {
            x: nvptx_block_idx_x(),
            y: nvptx_block_idx_y(),
            z: nvptx_block_idx_z(),
        }
    }
}

pub fn grid_dim() -> Dim3 {
    unsafe {
        Dim3 {
            x: nvptx_grid_dim_x(),
            y: nvptx_grid_dim_y(),
            z: nvptx_grid_dim_z(),
        }
    }
}

pub fn thread_idx() -> Idx3 {
    unsafe {
        Idx3 {
            x: nvptx_thread_idx_x(),
            y: nvptx_thread_idx_y(),
            z: nvptx_thread_idx_z(),
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

extern "platform-intrinsic" {
    pub fn nvptx_block_dim_x() -> i32;
    pub fn nvptx_block_dim_y() -> i32;
    pub fn nvptx_block_dim_z() -> i32;
    pub fn nvptx_block_idx_x() -> i32;
    pub fn nvptx_block_idx_y() -> i32;
    pub fn nvptx_block_idx_z() -> i32;
    pub fn nvptx_grid_dim_x() -> i32;
    pub fn nvptx_grid_dim_y() -> i32;
    pub fn nvptx_grid_dim_z() -> i32;
    pub fn nvptx_syncthreads() -> ();
    pub fn nvptx_thread_idx_x() -> i32;
    pub fn nvptx_thread_idx_y() -> i32;
    pub fn nvptx_thread_idx_z() -> i32;
}

pub fn index() -> usize {
    let block_id = block_idx().into_id(grid_dim());
    let thread_id = thread_idx().into_id(block_dim());
    (block_id + thread_id) as usize
}
