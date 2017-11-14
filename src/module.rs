
use ffi::cuda::*;
use ffi::vector_types::*;
use error::*;

use std::ptr::null_mut;
use std::os::raw::c_char;
use std::str::FromStr;

#[derive(Debug)]
pub struct Module(CUmodule);

impl Module {
    pub fn load(filename: &str) -> Result<Self> {
        let filename = str2cstring(filename);
        let mut handle = null_mut();
        unsafe { cuModuleLoad(&mut handle as *mut CUmodule, filename.as_ptr()) }
            .check()?;
        Ok(Module(handle))
    }

    pub fn get_function(&self, name: &str) -> Result<Function> {
        let name = str2cstring(name);
        let mut f = null_mut();
        unsafe { cuModuleGetFunction(&mut f as *mut CUfunction, self.0, name.as_ptr()) }
            .check()?;
        Ok(Function(f))
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { cuModuleUnload(self.0) }.check().expect(
            "Failed to unload module",
        );
    }
}

#[derive(Debug)]
pub struct Function(CUfunction);

#[derive(Debug, Clone, Copy)]
pub struct Dim3(dim3);

impl Dim3 {
    pub fn x(x: u32) -> Self {
        Dim3(dim3 { x: x, y: 1, z: 1 })
    }

    pub fn xy(x: u32, y: u32) -> Self {
        Dim3(dim3 { x: x, y: y, z: 1 })
    }

    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Dim3(dim3 { x: x, y: y, z: z })
    }
}

#[derive(Debug, Clone, Copy, NewType)]
pub struct Block(Dim3);

#[derive(Debug, Clone, Copy, NewType)]
pub struct Grid(Dim3);

fn str2cstring(s: &str) -> Vec<c_char> {
    let cstr = String::from_str(s).unwrap() + "\0";
    cstr.into_bytes().into_iter().map(|c| c as c_char).collect()
}
