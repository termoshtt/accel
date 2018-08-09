//! Resource of CUDA middle-IR (PTX/cubin)
//!
//! This module includes a wrapper of `cuLink*` and `cuModule*`
//! in [CUDA Driver APIs](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html).

use error::*;
use ffi::cuda::*;

use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_uint, c_void};
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::str::FromStr;

use super::kernel::Kernel;

/// Option for JIT compile
///
/// A wrapper of `CUjit_option`
/// http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d
pub type JITOption = Option<HashMap<CUjit_option, *mut c_void>>;

/// Parse JIT option to use for `cuLink*` APIs
fn parse(option: &JITOption) -> (c_uint, Vec<CUjit_option>, Vec<*mut c_void>) {
    if let &Some(ref hmap) = option {
        let opt_name: Vec<_> = hmap.keys().cloned().collect();
        let opt_value = hmap.values().cloned().collect();
        (opt_name.len() as c_uint, opt_name, opt_value)
    } else {
        (0, Vec::new(), Vec::new())
    }
}

/// Represent the resource of CUDA middle-IR (PTX/cubin)
#[derive(Debug)]
pub enum Data {
    PTX(String),
    PTXFile(PathBuf),
    Cubin(Vec<u8>),
    CubinFile(PathBuf),
}

impl Data {
    /// Constructor for `Data::PTX`
    pub fn ptx(s: &str) -> Data {
        Data::PTX(s.to_owned())
    }

    /// Constructor for `Data::Cubin`
    pub fn cubin(sl: &[u8]) -> Data {
        Data::Cubin(sl.to_vec())
    }

    /// Constructor for `Data::PTXFile`
    pub fn ptx_file(path: &Path) -> Data {
        Data::PTXFile(path.to_owned())
    }

    /// Constructor for `Data::CubinFile`
    pub fn cubin_file(path: &Path) -> Data {
        Data::CubinFile(path.to_owned())
    }
}

impl Data {
    /// Get type of PTX/cubin
    fn input_type(&self) -> CUjitInputType {
        match *self {
            Data::PTX(_) | Data::PTXFile(_) => CUjitInputType_enum::CU_JIT_INPUT_PTX,
            Data::Cubin(_) | Data::CubinFile(_) => CUjitInputType_enum::CU_JIT_INPUT_CUBIN,
        }
    }
}

/// OOP-like wrapper for `cuLink*` APIs
#[derive(Debug)]
pub struct Linker(CUlinkState);

impl Drop for Linker {
    fn drop(&mut self) {
        unsafe { cuLinkDestroy(self.0) }
            .check()
            .expect("Failed to release Linker");
    }
}

/// Link PTX/cubin into a module
pub fn link(data: &[Data], opt: &JITOption) -> Result<Module> {
    let mut l = Linker::create(opt)?;
    for d in data {
        l.add(d, opt)?;
    }
    let cubin = l.complete()?;
    Module::load(&cubin)
}

impl Linker {
    /// Create a new Linker
    pub fn create(option: &JITOption) -> Result<Self> {
        let (n, mut opt, mut opts) = parse(option);
        let mut st = null_mut();
        unsafe { cuLinkCreate_v2(n, opt.as_mut_ptr(), opts.as_mut_ptr(), &mut st as *mut _) }.check()?;
        Ok(Linker(st))
    }

    /// Wrapper of cuLinkAddData
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

    /// Wrapper of cuLinkAddFile
    unsafe fn add_file(&mut self, input_type: CUjitInputType, path: &Path, opt: &JITOption) -> Result<()> {
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

    /// Add a resouce into the linker stack.
    pub fn add(&mut self, data: &Data, opt: &JITOption) -> Result<()> {
        match *data {
            Data::PTX(ref ptx) => unsafe {
                let mut cstr = str2cstring(ptx);
                let ptr = cstr.as_mut_ptr() as *mut _;
                let n = cstr.len();
                self.add_data(data.input_type(), ptr, n, opt)?;
            },
            Data::Cubin(ref bin) => unsafe {
                let ptr = bin.as_ptr() as *mut _;
                let n = bin.len();
                self.add_data(data.input_type(), ptr, n, opt)?;
            },
            Data::PTXFile(ref path) | Data::CubinFile(ref path) => unsafe {
                self.add_file(data.input_type(), path, opt)?;
            },
        };
        Ok(())
    }

    /// Wrapper of cuLinkComplete
    ///
    /// LinkComplete returns a reference to cubin,
    /// which is managed by LinkState.
    /// Use owned strategy to avoid considering lifetime.
    pub fn complete(&mut self) -> Result<Data> {
        let mut cb = null_mut();
        unsafe {
            cuLinkComplete(self.0, &mut cb as *mut _, null_mut()).check()?;
            Ok(Data::cubin(CStr::from_ptr(cb as _).to_bytes()))
        }
    }
}

/// OOP-like wrapper of `cuModule*` APIs
#[derive(Debug)]
pub struct Module(CUmodule);

impl Module {
    /// integrated loader of Data
    pub fn load(data: &Data) -> Result<Self> {
        match *data {
            Data::PTX(ref ptx) => unsafe {
                let cstr = str2cstring(ptx);
                Self::load_data(cstr.as_ptr() as _)
            },
            Data::Cubin(ref bin) => unsafe {
                let ptr = bin.as_ptr() as *mut _;
                Self::load_data(ptr)
            },
            Data::PTXFile(ref path) | Data::CubinFile(ref path) => Self::load_file(path),
        }
    }

    /// Wrapper for `cuModuleLoadData`
    unsafe fn load_data(ptr: *const c_void) -> Result<Self> {
        let mut handle = null_mut();
        let m = &mut handle as *mut CUmodule;
        cuModuleLoadData(m, ptr).check()?;
        Ok(Module(handle))
    }

    /// Wrapper for `cuModuleLoad`
    pub fn load_file(path: &Path) -> Result<Self> {
        let mut handle = null_mut();
        let m = &mut handle as *mut CUmodule;
        let filename = str2cstring(path.to_str().unwrap());
        unsafe { cuModuleLoad(m, filename.as_ptr()) }.check()?;
        Ok(Module(handle))
    }

    pub fn from_str(ptx: &str) -> Result<Self> {
        let data = Data::ptx(ptx);
        Self::load(&data)
    }

    /// Wrapper of `cuModuleGetFunction`
    pub fn get_kernel<'m>(&'m self, name: &str) -> Result<Kernel<'m>> {
        let name = str2cstring(name);
        let mut func = null_mut();
        unsafe { cuModuleGetFunction(&mut func as *mut CUfunction, self.0, name.as_ptr()) }.check()?;
        Ok(Kernel { func, _m: self })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { cuModuleUnload(self.0) }
            .check()
            .expect("Failed to unload module");
    }
}

fn str2cstring(s: &str) -> Vec<c_char> {
    let cstr = String::from_str(s).unwrap() + "\0";
    cstr.into_bytes().into_iter().map(|c| c as c_char).collect()
}
