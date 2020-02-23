//! Execution control in
//! [CUDA Deriver APIs](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)

use crate::{error::*, ffi_call};
use anyhow::Result;
use cuda::*;
use cudart::*;

use std::ops::Deref;
use std::os::raw::*;
use std::ptr::null_mut;

use super::module::*;

/// CUDA Kernel function
#[derive(Debug)]
pub struct Kernel<'m> {
    pub(crate) func: CUfunction,
    pub(crate) _m: &'m Module,
}

impl<'m> Kernel<'m> {
    /// Launch CUDA kernel using `cuLaunchKernel`
    pub unsafe fn launch(
        &mut self,
        args: *mut *mut c_void,
        grid: Grid,
        block: Block,
    ) -> Result<()> {
        Ok(ffi_call!(
            cuLaunchKernel,
            self.func,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            0,          /* FIXME: no shared memory */
            null_mut(), /* use default stream */
            args,
            null_mut() /* no extra */
        )?)
    }
}

/// Get type-erased pointer
///
/// ```
/// # use accel::driver::kernel::void_cast;
/// let a = 1_usize;
/// let p = void_cast(&a);
/// unsafe { assert_eq!(*(p as *mut usize), 1) };
/// ```
///
/// This returns the pointer for slice, and the length of slice is dropped:
///
/// ```
/// # use accel::driver::kernel::void_cast;
/// # use std::os::raw::c_void;
/// let s: &[f64] = &[0.0; 4];
/// let p = s.as_ptr() as *mut c_void;
/// let p1 = void_cast(s);
/// let p2 = void_cast(&s);
/// assert_eq!(p, p1);
/// assert_ne!(p, p2); // Result of slice and &slice are different!
/// ```
pub fn void_cast<T: ?Sized>(r: &T) -> *mut c_void {
    &*r as *const T as *mut c_void
}

/// Size of Block (thread block) in [CUDA thread hierarchy]( http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model )
#[derive(Debug, Clone, Copy)]
pub struct Block(dim3);

impl Deref for Block {
    type Target = dim3;
    fn deref(&self) -> &dim3 {
        &self.0
    }
}

impl Block {
    /// one-dimensional
    pub fn x(x: u32) -> Self {
        Block(dim3 { x: x, y: 1, z: 1 })
    }

    /// two-dimensional
    pub fn xy(x: u32, y: u32) -> Self {
        Block(dim3 { x: x, y: y, z: 1 })
    }

    /// three-dimensional
    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Block(dim3 { x: x, y: y, z: z })
    }
}

/// Size of Grid (grid of blocks) in [CUDA thread hierarchy]( http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model )
#[derive(Debug, Clone, Copy)]
pub struct Grid(dim3);

impl Deref for Grid {
    type Target = dim3;
    fn deref(&self) -> &dim3 {
        &self.0
    }
}

impl Grid {
    /// one-dimensional
    pub fn x(x: u32) -> Self {
        Grid(dim3 { x: x, y: 1, z: 1 })
    }

    /// two-dimensional
    pub fn xy(x: u32, y: u32) -> Self {
        Grid(dim3 { x: x, y: y, z: 1 })
    }

    /// three-dimensional
    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Grid(dim3 { x: x, y: y, z: z })
    }
}
