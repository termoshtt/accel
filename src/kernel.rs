
use ffi::cuda::*;
use ffi::vector_types::*;
use error::*;

use std::ptr::null_mut;
use std::os::raw::{c_char, c_void};
use std::str::FromStr;
use std::path::Path;

#[derive(Debug)]
pub struct PTXModule(CUmodule);

impl PTXModule {
    pub fn from_str(ptx_str: &str) -> Result<Self> {
        let ptx = str2cstring(ptx_str);
        let mut handle = null_mut();
        unsafe { cuModuleLoadData(&mut handle as *mut CUmodule, ptx.as_ptr() as *mut c_void) }
            .check()?;
        Ok(PTXModule(handle))
    }

    pub fn load(filename: &str) -> Result<Self> {
        if !Path::new(filename).exists() {
            panic!("File not found: {}", filename);
        }
        let filename = str2cstring(filename);
        let mut handle = null_mut();
        unsafe { cuModuleLoad(&mut handle as *mut CUmodule, filename.as_ptr()) }
            .check()?;
        Ok(PTXModule(handle))
    }

    pub fn get_kernel<'m>(&'m self, name: &str) -> Result<Kernel<'m>> {
        let name = str2cstring(name);
        let mut func = null_mut();
        unsafe { cuModuleGetFunction(&mut func as *mut CUfunction, self.0, name.as_ptr()) }
            .check()?;
        Ok(Kernel { func, _m: self })
    }
}

impl Drop for PTXModule {
    fn drop(&mut self) {
        unsafe { cuModuleUnload(self.0) }.check().expect(
            "Failed to unload module",
        );
    }
}

#[derive(Debug)]
pub struct Kernel<'m> {
    func: CUfunction,
    _m: &'m PTXModule,
}

impl<'m> Kernel<'m> {
    pub unsafe fn launch(
        &mut self,
        args: *mut *mut c_void,
        grid: Grid,
        block: Block,
    ) -> Result<()> {
        cuLaunchKernel(
            self.func,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            0, // FIXME: no shared memory
            null_mut(), // use default stream
            args,
            null_mut(), // no extra
        ).check()
    }
}

pub fn void_cast<T>(r: &T) -> *mut c_void {
    &*r as *const T as *mut c_void
}

#[derive(Debug, Clone, Copy, NewType)]
pub struct Block(dim3);

impl Block {
    pub fn x(x: u32) -> Self {
        Block(dim3 { x: x, y: 1, z: 1 })
    }

    pub fn xy(x: u32, y: u32) -> Self {
        Block(dim3 { x: x, y: y, z: 1 })
    }

    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Block(dim3 { x: x, y: y, z: z })
    }
}

#[derive(Debug, Clone, Copy, NewType)]
pub struct Grid(dim3);

impl Grid {
    pub fn x(x: u32) -> Self {
        Grid(dim3 { x: x, y: 1, z: 1 })
    }

    pub fn xy(x: u32, y: u32) -> Self {
        Grid(dim3 { x: x, y: y, z: 1 })
    }

    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Grid(dim3 { x: x, y: y, z: z })
    }
}

fn str2cstring(s: &str) -> Vec<c_char> {
    let cstr = String::from_str(s).unwrap() + "\0";
    cstr.into_bytes().into_iter().map(|c| c as c_char).collect()
}
