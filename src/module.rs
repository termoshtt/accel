
use ffi::cuda::*;
use error::*;

use std::ptr::null_mut;
use std::os::raw::c_char;
use std::str::FromStr;

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

pub struct Function(CUfunction);

fn str2cstring(s: &str) -> Vec<c_char> {
    let cstr = String::from_str(s).unwrap() + "\0";
    cstr.into_bytes().into_iter().map(|c| c as c_char).collect()
}
