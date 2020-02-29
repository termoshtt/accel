//! Resource of CUDA middle-IR (PTX/cubin)
//!
//! This module includes a wrapper of `cuLink*` and `cuModule*`
//! in [CUDA Driver APIs](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html).

use super::{context::*, instruction::*};
use crate::{error::*, ffi_call, ffi_call_unsafe, ffi_new_unsafe};
use anyhow::{ensure, Result};
use cuda::*;
use cudart::*;
use std::{ffi::*, ops::Deref, path::Path, ptr::null_mut};

/// CUDA Kernel function
#[derive(Debug)]
pub struct Kernel<'m> {
    func: CUfunction,
    _m: &'m Module<'m>,
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
/// # use accel::driver::module::void_cast;
/// let a = 1_usize;
/// let p = void_cast(&a);
/// unsafe { assert_eq!(*(p as *mut usize), 1) };
/// ```
///
/// This returns the pointer for slice, and the length of slice is dropped:
///
/// ```
/// # use accel::driver::module::void_cast;
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
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
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
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
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

/// OOP-like wrapper of `cuModule*` APIs
#[derive(Debug)]
pub struct Module<'ctx> {
    module: CUmodule,
    context: &'ctx Context,
}

impl<'ctx> Drop for Module<'ctx> {
    fn drop(&mut self) {
        ffi_call_unsafe!(cuModuleUnload, self.module).expect("Failed to unload module");
    }
}

impl<'ctx> Module<'ctx> {
    /// integrated loader of Instruction
    pub fn load(context: &'ctx Context, data: &Instruction) -> Result<Self> {
        ensure!(context.is_current()?, "Given context is not current");
        match *data {
            Instruction::PTX(ref ptx) => {
                let module = ffi_new_unsafe!(cuModuleLoadData, ptx.as_ptr() as *const _)?;
                Ok(Module { module, context })
            }
            Instruction::Cubin(ref bin) => {
                let module = ffi_new_unsafe!(cuModuleLoadData, bin.as_ptr() as *const _)?;
                Ok(Module { module, context })
            }
            Instruction::PTXFile(ref path) | Instruction::CubinFile(ref path) => {
                let filename = path_to_cstring(path);
                let module = ffi_new_unsafe!(cuModuleLoad, filename.as_ptr())?;
                Ok(Module { module, context })
            }
        }
    }

    pub fn from_str(context: &'ctx Context, ptx: &str) -> Result<Self> {
        let data = Instruction::ptx(ptx);
        Self::load(context, &data)
    }

    /// Wrapper of `cuModuleGetFunction`
    pub fn get_kernel<'m>(&'m self, name: &str) -> Result<Kernel<'m>> {
        ensure!(self.context.is_current()?, "Given context is not current");
        let name = CString::new(name).expect("Invalid Kernel name");
        let func = ffi_new_unsafe!(cuModuleGetFunction, self.module, name.as_ptr())?;
        Ok(Kernel { func, _m: self })
    }
}

fn path_to_cstring(path: &Path) -> CString {
    CString::new(path.to_str().unwrap()).expect("Invalid Path")
}

#[cfg(test)]
mod tests {
    use super::super::device::*;
    use super::*;

    #[test]
    fn load_do_nothing() -> anyhow::Result<()> {
        // generated by do_nothing example in accel-derive
        let ptx = r#"
        .version 3.2
        .target sm_30
        .address_size 64
        .visible .entry do_nothing()
        {
          ret;
        }
        "#;
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let _mod = Module::from_str(&ctx, ptx)?;
        Ok(())
    }
}
