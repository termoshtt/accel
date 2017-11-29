
use ffi::cuda::*;
use ffi::vector_types::*;
use error::*;

use std::ptr::null_mut;
use std::os::raw::{c_uint, c_char, c_void};
use std::str::FromStr;
use std::path::{Path, PathBuf};
use std::collections::HashMap;

pub type JITOption = Option<HashMap<CUjit_option, *mut c_void>>;

fn parse(option: &JITOption) -> (c_uint, Vec<CUjit_option>, Vec<*mut c_void>) {
    if let &Some(ref hmap) = option {
        let opt_name: Vec<_> = hmap.keys().cloned().collect();
        let opt_value = hmap.values().cloned().collect();
        (opt_name.len() as c_uint, opt_name, opt_value)
    } else {
        (0, Vec::new(), Vec::new())
    }
}

pub enum Data {
    PTX(String),
    PTXFile(PathBuf),
    CubinFile(PathBuf),
}

impl Data {
    fn input_type(&self) -> CUjitInputType {
        match *self {
            Data::PTX(_) | Data::PTXFile(_) => CUjitInputType_enum::CU_JIT_INPUT_PTX,
            Data::CubinFile(_) => CUjitInputType_enum::CU_JIT_INPUT_CUBIN,
        }
    }
}

#[derive(Debug)]
pub struct Linker(CUlinkState);

impl Drop for Linker {
    fn drop(&mut self) {
        unsafe { cuLinkDestroy(self.0) }.check().expect(
            "Failed to release Linker",
        );
    }
}

impl Linker {
    pub fn create(option: &JITOption) -> Result<Self> {
        let (n, mut opt, mut opts) = parse(option);
        let mut st = null_mut();
        unsafe { cuLinkCreate_v2(n, opt.as_mut_ptr(), opts.as_mut_ptr(), &mut st as *mut _) }
            .check()?;
        Ok(Linker(st))
    }

    unsafe fn add_data(
        &mut self,
        input_type: CUjitInputType,
        data: *mut c_void,
        size: usize,
        opt: &JITOption,
    ) -> Result<()> {
        let (nopts, mut opts, mut opt_vals) = parse(opt);
        let name = str2cstring(&"\0");
        cuLinkAddData_v2(
            self.0,
            input_type,
            data,
            size,
            name.as_ptr(),
            nopts,
            opts.as_mut_ptr(),
            opt_vals.as_mut_ptr(),
        ).check()?;
        Ok(())
    }

    unsafe fn add_file(
        &mut self,
        input_type: CUjitInputType,
        path: &Path,
        opt: &JITOption,
    ) -> Result<()> {
        let filename = str2cstring(path.to_str().unwrap()).as_mut_ptr();
        let (nopts, mut opts, mut opt_vals) = parse(opt);
        cuLinkAddFile_v2(
            self.0,
            input_type,
            filename,
            nopts,
            opts.as_mut_ptr(),
            opt_vals.as_mut_ptr(),
        ).check()?;
        Ok(())
    }

    pub fn add(&mut self, data: &Data, opt: &JITOption) -> Result<()> {
        match *data {
            Data::PTX(ref ptx) => unsafe {
                let (ptr, n) = str2void(ptx);
                self.add_data(data.input_type(), ptr, n, opt)?;
            },
            Data::PTXFile(ref path) |
            Data::CubinFile(ref path) => unsafe {
                self.add_file(data.input_type(), path, opt)?;
            },
        };
        Ok(())
    }
}

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

unsafe fn str2void(s: &str) -> (*mut c_void, usize) {
    let mut cstr = str2cstring(s);
    (cstr.as_mut_ptr() as *mut _, cstr.len())
}
