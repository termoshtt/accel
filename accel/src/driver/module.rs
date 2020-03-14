//! Resource of CUDA middle-IR (PTX/cubin)
//!
//! This module includes a wrapper of `cuLink*` and `cuModule*`
//! in [CUDA Driver APIs](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html).

use super::{context::*, instruction::*, *};
use crate::{error::*, ffi_call, ffi_call_unsafe, ffi_new_unsafe};
use anyhow::{ensure, Result};
use cuda::*;
use std::{ffi::*, path::Path, ptr::null_mut};

/// CUDA Kernel function
#[derive(Debug)]
pub struct Kernel<'ctx> {
    func: CUfunction,
    _m: &'ctx Module<'ctx>,
}

impl<'ctx> Kernel<'ctx> {
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

/// Type which can be sent to the device as kernel argument
///
/// Pointer
/// --------
///
/// ```
/// # use accel::driver::module::*;
/// # use std::ffi::*;
/// let a: i32 = 10;
/// let p = &a as *const i32;
/// assert_eq!(DeviceSend::as_ptr(&p), p as *mut c_void);
/// ```
pub trait DeviceSend: Sized {
    fn as_ptr(&self) -> *mut c_void;
}

impl<T> DeviceSend for *mut T {
    fn as_ptr(&self) -> *mut c_void {
        *self as *mut c_void
    }
}

impl<T> DeviceSend for *const T {
    fn as_ptr(&self) -> *mut c_void {
        *self as *mut c_void
    }
}

macro_rules! impl_device_send {
    ($target:ty) => {
        impl DeviceSend for $target {
            fn as_ptr(&self) -> *mut c_void {
                self as *const $target as *mut c_void
            }
        }
    };
}

impl_device_send!(bool);
impl_device_send!(i8);
impl_device_send!(i16);
impl_device_send!(i32);
impl_device_send!(i64);
impl_device_send!(isize);
impl_device_send!(u8);
impl_device_send!(u16);
impl_device_send!(u32);
impl_device_send!(u64);
impl_device_send!(usize);
impl_device_send!(f32);
impl_device_send!(f64);

/// Arbitary number of tuple of kernel arguments
///
/// ```
/// # use accel::driver::module::*;
/// # use std::ffi::*;
/// let a: i32 = 10;
/// let b: f32 = 1.0;
/// assert_eq!(
///   KernelParameters::kernel_params(&(&a, &b)),
///   vec![&a as *const i32 as *mut _, &b as *const f32 as *mut _, ]
/// );
/// ```
pub trait KernelParameters<'arg> {
    /// Get a list of kernel parameters to be passed into [cuLaunchKernel]
    ///
    /// [cuLaunchKernel]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15
    fn kernel_params(&self) -> Vec<*mut c_void>;
}

macro_rules! impl_kernel_parameters {
    ($($name:ident),*; $($num:tt),*) => {
        impl<'arg, $($name : DeviceSend),*> KernelParameters<'arg> for ($( &'arg $name, )*) {
            fn kernel_params(&self) -> Vec<*mut c_void> {
                vec![$( self.$num.as_ptr() ),*]
            }
        }
    }
}

impl_kernel_parameters!(;);
impl_kernel_parameters!(D0; 0);
impl_kernel_parameters!(D0, D1; 0, 1);
impl_kernel_parameters!(D0, D1, D2; 0, 1, 2);
impl_kernel_parameters!(D0, D1, D2, D3; 0, 1, 2, 3);
impl_kernel_parameters!(D0, D1, D2, D3, D4; 0, 1, 2, 3, 4);
impl_kernel_parameters!(D0, D1, D2, D3, D4, D5; 0, 1, 2, 3, 4, 5);
impl_kernel_parameters!(D0, D1, D2, D3, D4, D5, D6; 0, 1, 2, 3, 4, 5, 6);
impl_kernel_parameters!(D0, D1, D2, D3, D4, D5, D6, D7; 0, 1, 2, 3, 4, 5, 6, 7);
impl_kernel_parameters!(D0, D1, D2, D3, D4, D5, D6, D7, D8; 0, 1, 2, 3, 4, 5, 6, 7, 8);
impl_kernel_parameters!(D0, D1, D2, D3, D4, D5, D6, D7, D8, D9; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl_kernel_parameters!(D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
impl_kernel_parameters!(D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11);

/// CUDA Kernel launcher trait
pub trait Launchable<'arg> {
    type Args: KernelParameters<'arg>;
    fn get_kernel(&self) -> Result<Kernel>;
    fn launch(&self, grid: Grid, block: Block, args: Self::Args) -> Result<()> {
        let mut params = args.kernel_params();
        Ok(ffi_call_unsafe!(
            cuLaunchKernel,
            self.get_kernel()?.func,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            0,          /* FIXME: no shared memory */
            null_mut(), /* use default stream */
            params.as_mut_ptr(),
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
    pub fn get_kernel(&self, name: &str) -> Result<Kernel> {
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
