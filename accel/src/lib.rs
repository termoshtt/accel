//! GPGPU framework for Rust

extern crate cuda_driver_sys as cuda;

pub mod array;
pub mod device;
pub mod error;
pub mod instruction;
pub mod linker;
pub mod memory;
pub mod module;

use std::sync::Once;

pub use device::Device;

/// Initializer for CUDA Driver API
static DRIVER_API_INIT: Once = Once::new();
fn cuda_driver_init() {
    DRIVER_API_INIT.call_once(|| {
        ffi_call_unsafe!(cuda::cuInit, 0).expect("Initialization of CUDA Driver API failed");
    })
}

/// Size of Block (thread block) in [CUDA thread hierarchy]( http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model )
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Block {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Block {
    /// one-dimensional
    pub fn x(x: u32) -> Self {
        Block { x: x, y: 1, z: 1 }
    }

    /// two-dimensional
    pub fn xy(x: u32, y: u32) -> Self {
        Block { x: x, y: y, z: 1 }
    }

    /// three-dimensional
    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Block { x: x, y: y, z: z }
    }
}

/// Size of Grid (grid of blocks) in [CUDA thread hierarchy]( http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model )
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Grid {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Grid {
    /// one-dimensional
    pub fn x(x: u32) -> Self {
        Grid { x: x, y: 1, z: 1 }
    }

    /// two-dimensional
    pub fn xy(x: u32, y: u32) -> Self {
        Grid { x: x, y: y, z: 1 }
    }

    /// three-dimensional
    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Grid { x: x, y: y, z: z }
    }
}
