//! Resource of CUDA middle-IR (PTX/cubin)
//!
//! This module includes a wrapper of `cuLink*` and `cuModule*`
//! in [CUDA Driver APIs](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html).

use super::{cuda_driver_init, module::*};
use crate::{error::*, ffi_call, ffi_call_unsafe};
use anyhow::{ensure, Result};
use cuda::*;
use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    mem::MaybeUninit,
    os::raw::c_void,
    path::{Path, PathBuf},
    ptr::null_mut,
};

/// Configure generator for [CUjit_option] required in `cuLink*` APIs
///
/// [CUjit_option]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d
#[derive(Debug, Clone, Default)]
pub struct JITConfig {
    /// CU_JIT_MAX_REGISTERS, Applies to compiler only
    ///
    /// - Max number of registers that a thread may use.
    pub max_registers: Option<u32>,

    /// CU_JIT_THREADS_PER_BLOCK, Applies to compiler only
    ///
    /// - **IN**: Specifies minimum number of threads per block to target compilation for
    /// - **OUT**: Returns the number of threads the compiler actually targeted.
    ///   This restricts the resource utilization fo the compiler (e.g. max registers) such that a block with the given number of threads should be able to launch based on register limitations.
    ///
    /// Note
    /// ----
    /// This option does not currently take into account any other resource limitations, such as shared memory utilization. Cannot be combined with CU_JIT_TARGET.
    pub threads_per_block: Option<u32>,

    /// CU_JIT_WALL_TIME, Applies to compiler and linker
    ///
    /// - Overwrites the option value with the total wall clock time, in milliseconds, spent in the compiler and linker
    /// - Option type: float
    pub wall_time: Option<f32>,

    /// CU_JIT_INFO_LOG_BUFFER, Applies to compiler and linker
    ///
    /// - Pointer to a buffer in which to print any log messages that are informational in nature (the buffer size is specified via option CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)
    ///
    /// CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, Applies to compiler and linker
    ///
    /// - **IN**: Log buffer size in bytes. Log messages will be capped at this size (including null terminator)
    /// - **OUT**: Amount of log buffer filled with messages
    pub info_log_buffer: Option<CString>,

    /// CU_JIT_ERROR_LOG_BUFFER, Applies to compiler and linker
    ///
    /// - Pointer to a buffer in which to print any log messages that reflect errors (the buffer size is specified via option CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
    ///
    /// CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, Applies to compiler and linker
    ///
    /// - **IN**: Log buffer size in bytes. Log messages will be capped at this size (including null terminator)
    /// - **OUT**: Amount of log buffer filled with messages
    pub error_log_buffer: Option<CString>,

    /// CU_JIT_OPTIMIZATION_LEVEL, Applies to compiler only
    ///
    /// - Level of optimizations to apply to generated code (0 - 4), with 4 being the default and highest level of optimizations.
    pub optimization_level: Option<u32>,

    /// CU_JIT_TARGET_FROM_CUCONTEXT, Applies to compiler and linker
    ///
    /// - No option value required. Determines the target based on the current attached context (default)
    pub target_from_cucontext: Option<()>,

    /// CU_JIT_TARGET, Applies to compiler and linker
    ///
    /// - Target is chosen based on supplied CUjit_target. Cannot be combined with CU_JIT_THREADS_PER_BLOCK.
    pub target: Option<CUjit_target>,

    /// CU_JIT_FALLBACK_STRATEGY, Applies to compiler only
    ///
    /// - Specifies choice of fallback strategy if matching cubin is not found. Choice is based on supplied CUjit_fallback.
    ///   This option cannot be used with cuLink* APIs as the linker requires exact matches.
    pub fallback_strategy: Option<CUjit_fallback>,

    /// CU_JIT_GENERATE_DEBUG_INFO, Applies to compiler and linker
    ///
    /// - Specifies whether to create debug information in output (-g) (0: false, default)
    pub generate_debug_info: Option<i32>,

    /// CU_JIT_LOG_VERBOSE, Applies to compiler and linker
    ///
    /// - Generate verbose log messages (0: false, default)
    pub log_verbose: Option<i32>,

    /// CU_JIT_GENERATE_LINE_INFO, Applies to compiler only
    ///
    /// - Generate line number information (-lineinfo) (0: false, default)
    pub generate_line_info: Option<i32>,

    /// CU_JIT_CACHE_MODE, Applies to compiler only
    ///
    /// - Specifies whether to enable caching explicitly (-dlcm) Choice is based on supplied CUjit_cacheMode_enum.
    pub cache_mode: Option<CUjit_cacheMode_enum>,

    /// CU_JIT_NEW_SM3X_OPT
    ///
    /// - The below jit options are used for internal purposes only, in this version of CUDA
    pub new_sm3x_opt: Option<u32>,

    /// CU_JIT_FAST_COMPILE
    pub fast_compile: bool,

    /// CU_JIT_GLOBAL_SYMBOL_NAMES, Applies to dynamic linker only
    ///
    /// - Array of device symbol names that will be relocated to the corresponing host addresses stored in CU_JIT_GLOBAL_SYMBOL_ADDRESSES.
    ///   Must contain CU_JIT_GLOBAL_SYMBOL_COUNT entries. When loding a device module, driver will relocate all encountered unresolved symbols to the host addresses.
    ///   It is only allowed to register symbols that correspond to unresolved global variables. It is illegal to register the same device symbol at multiple addresses.
    ///
    /// CU_JIT_GLOBAL_SYMBOL_ADDRESSES, Applies to dynamic linker only
    ///
    /// - Array of host addresses that will be used to relocate corresponding device symbols stored in CU_JIT_GLOBAL_SYMBOL_NAMES.
    ///   Must contain CU_JIT_GLOBAL_SYMBOL_COUNT entries.
    ///
    /// CU_JIT_GLOBAL_SYMBOL_COUNT, Applies to dynamic linker only
    ///
    /// - Number of entries in CU_JIT_GLOBAL_SYMBOL_NAMES and CU_JIT_GLOBAL_SYMBOL_ADDRESSES arrays.
    pub global_symbol: HashMap<CString, *mut c_void>,
}

impl JITConfig {
    /// Pack configure into C API compatible format
    fn pack(&self) -> (u32, Vec<CUjit_option>, Vec<*mut c_void>) {
        // TODO
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
    pub fn ptx_file(path: &Path) -> Result<Self> {
        ensure!(path.exists(), "PTX file does not found: {}", path.display());
        Ok(Data::PTXFile(path.to_owned()))
    }

    /// Constructor for `Data::CubinFile`
    pub fn cubin_file(path: &Path) -> Result<Self> {
        ensure!(
            path.exists(),
            "cubin file does not found: {}",
            path.display()
        );
        Ok(Data::CubinFile(path.to_owned()))
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

/// Consuming builder for cubin from PTX and cubins
pub struct Linker {
    state: CUlinkState,
    cfg: JITConfig,
}

impl Drop for Linker {
    fn drop(&mut self) {
        ffi_call_unsafe!(cuLinkDestroy, self.state).expect("Failed to release Linker");
    }
}

impl Linker {
    /// Create a new Linker
    pub fn create(cfg: JITConfig) -> Result<Self> {
        cuda_driver_init();
        let (n, mut opt, mut opts) = cfg.pack();
        let state = unsafe {
            let mut state = MaybeUninit::uninit();
            ffi_call!(
                cuLinkCreate_v2,
                n,
                opt.as_mut_ptr(),
                opts.as_mut_ptr(),
                state.as_mut_ptr()
            )?;
            state.assume_init()
        };
        Ok(Linker { state, cfg })
    }

    /// Wrapper of cuLinkAddData
    unsafe fn add_data(self, input_type: CUjitInputType, data: &[u8]) -> Result<Self> {
        let (nopts, mut opts, mut opt_vals) = self.cfg.pack();
        let name = CString::new("").unwrap();
        ffi_call!(
            cuLinkAddData_v2,
            self.state,
            input_type,
            data.as_ptr() as *mut _,
            data.len(),
            name.as_ptr(),
            nopts,
            opts.as_mut_ptr(),
            opt_vals.as_mut_ptr()
        )?;
        Ok(self)
    }

    /// Wrapper of cuLinkAddFile
    unsafe fn add_file(self, input_type: CUjitInputType, path: &Path) -> Result<Self> {
        let filename = CString::new(path.to_str().unwrap()).expect("Invalid file path");
        let (nopts, mut opts, mut opt_vals) = self.cfg.pack();
        ffi_call!(
            cuLinkAddFile_v2,
            self.state,
            input_type,
            filename.as_ptr(),
            nopts,
            opts.as_mut_ptr(),
            opt_vals.as_mut_ptr()
        )?;
        Ok(self)
    }

    /// Add a resouce into the linker stack.
    pub fn add(self, data: &Data) -> Result<Self> {
        Ok(match *data {
            Data::PTX(ref ptx) => unsafe {
                let cstr = CString::new(ptx.as_bytes()).expect("Invalid PTX String");
                self.add_data(data.input_type(), cstr.as_bytes_with_nul())?
            },
            Data::Cubin(ref bin) => unsafe { self.add_data(data.input_type(), &bin)? },
            Data::PTXFile(ref path) | Data::CubinFile(ref path) => unsafe {
                self.add_file(data.input_type(), path)?
            },
        })
    }

    /// Wrapper of cuLinkComplete
    ///
    /// LinkComplete returns a reference to cubin,
    /// which is managed by LinkState.
    /// Use owned strategy to avoid considering lifetime.
    pub fn complete(self) -> Result<Data> {
        let mut cb = null_mut();
        unsafe {
            ffi_call!(cuLinkComplete, self.state, &mut cb as *mut _, null_mut())?;
            Ok(Data::cubin(CStr::from_ptr(cb as _).to_bytes()))
        }
    }
}

/// Link PTX/cubin into a module
pub fn link(data: &[Data], cfg: JITConfig) -> Result<Module> {
    let mut l = Linker::create(cfg)?;
    for d in data {
        l = l.add(d)?;
    }
    let cubin = l.complete()?;
    Module::load(&cubin)
}

#[cfg(test)]
mod tests {
    use super::super::device::*;
    use super::*;

    #[test]
    fn create() -> Result<()> {
        let device = Device::nth(0)?;
        let _ctx = device.create_context_auto()?;

        let jit_cfg = JITConfig::default();
        let _linker = Linker::create(jit_cfg)?;
        Ok(())
    }

    #[test]
    fn ptx_file() -> Result<()> {
        let device = Device::nth(0)?;
        let _ctx = device.create_context_auto()?;

        let jit_cfg = JITConfig::default();
        let linker = Linker::create(jit_cfg)?;
        let data = Data::ptx_file(Path::new("tests/data/add.ptx"))?;
        linker.add(&data)?;
        Ok(())
    }

    #[ignore] // FIXME Causes CUDA_ERROR_NO_BINARY_FOR_GPU
    #[test]
    fn cubin_file() -> Result<()> {
        let device = Device::nth(0)?;
        let _ctx = device.create_context_auto()?;

        let jit_cfg = JITConfig::default();
        let linker = Linker::create(jit_cfg)?;
        let data = Data::cubin_file(Path::new("tests/data/add.cubin"))?;
        linker.add(&data)?;
        Ok(())
    }
}
