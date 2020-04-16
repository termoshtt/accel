//! CUDA JIT compiler and Linkers

use super::{device::*, module::*};
use crate::{contexted_call, error::*};
use cuda::*;
use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    mem::MaybeUninit,
    os::raw::c_void,
    path::Path,
    ptr::null_mut,
};

// TODO
#[derive(Debug, Clone)]
pub struct LogBuffer {}

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
    pub info_log_buffer: Option<LogBuffer>,

    /// CU_JIT_ERROR_LOG_BUFFER, Applies to compiler and linker
    ///
    /// - Pointer to a buffer in which to print any log messages that reflect errors (the buffer size is specified via option CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
    ///
    /// CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, Applies to compiler and linker
    ///
    /// - **IN**: Log buffer size in bytes. Log messages will be capped at this size (including null terminator)
    /// - **OUT**: Amount of log buffer filled with messages
    pub error_log_buffer: Option<LogBuffer>,

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
    fn pack(&mut self) -> (u32, Vec<CUjit_option>, Vec<*mut c_void>) {
        let mut opt_keys = Vec::new();
        let mut opt_values = Vec::new();

        macro_rules! check_option {
            ( $tag:ident, $opt_name:ident) => {
                if let Some($opt_name) = self.$opt_name.as_ref() {
                    opt_keys.push(CUjit_option::$tag);
                    opt_values.push($opt_name as *const _ as *mut c_void);
                }
            };
        }
        check_option!(CU_JIT_MAX_REGISTERS, max_registers);
        check_option!(CU_JIT_THREADS_PER_BLOCK, threads_per_block);
        check_option!(CU_JIT_WALL_TIME, wall_time);
        check_option!(CU_JIT_OPTIMIZATION_LEVEL, optimization_level);
        check_option!(CU_JIT_TARGET, target);
        check_option!(CU_JIT_FALLBACK_STRATEGY, fallback_strategy);
        check_option!(CU_JIT_GENERATE_DEBUG_INFO, generate_debug_info);
        check_option!(CU_JIT_LOG_VERBOSE, log_verbose);
        check_option!(CU_JIT_GENERATE_LINE_INFO, generate_line_info);
        check_option!(CU_JIT_CACHE_MODE, cache_mode);
        check_option!(CU_JIT_NEW_SM3X_OPT, new_sm3x_opt);

        if self.fast_compile {
            opt_keys.push(CUjit_option::CU_JIT_FAST_COMPILE);
            opt_values.push(&self.fast_compile as *const bool as *mut c_void);
        }

        if let Some(_info_log_buffer) = self.info_log_buffer.as_mut() {
            unimplemented!("Log for JIT is not supported yet");
        }

        if let Some(_error_log_buffer) = self.error_log_buffer.as_mut() {
            unimplemented!("Log for JIT is not supported yet");
        }

        if self.global_symbol.len() != 0 {
            unimplemented!("GLOBAL_SYMBOL flags are not supported yet");
        }
        assert_eq!(opt_keys.len(), opt_values.len());
        (opt_keys.len() as u32, opt_keys, opt_values)
    }
}

/// Consuming builder for cubin from PTX and cubins
pub struct Linker<'ctx> {
    state: CUlinkState,
    cfg: JITConfig,
    context: &'ctx Context,
}

impl<'ctx> Drop for Linker<'ctx> {
    fn drop(&mut self) {
        if let Err(e) = contexted_call!(self, cuLinkDestroy, self.state) {
            log::error!("Failed to release Linker: {:?}", e)
        }
    }
}

impl<'ctx> Contexted for Linker<'ctx> {
    fn get_context(&self) -> &Context {
        self.context
    }
}

impl<'ctx> Linker<'ctx> {
    /// Create a new Linker
    pub fn create(context: &'ctx Context, mut cfg: JITConfig) -> Result<Self> {
        let (n, mut opt, mut opts) = cfg.pack();
        #[allow(unused_unsafe)]
        let state = unsafe {
            let mut state = MaybeUninit::uninit();
            contexted_call!(
                context,
                cuLinkCreate_v2,
                n,
                opt.as_mut_ptr(),
                opts.as_mut_ptr(),
                state.as_mut_ptr()
            )?;
            state.assume_init()
        };
        Ok(Linker {
            state,
            cfg,
            context,
        })
    }

    /// Wrapper of cuLinkAddData
    #[allow(unused_unsafe)]
    unsafe fn add_data(mut self, input_type: CUjitInputType, data: &[u8]) -> Result<Self> {
        let (nopts, mut opts, mut opt_vals) = self.cfg.pack();
        let name = CString::new("").unwrap();
        {
            contexted_call!(
                &self,
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
        }
        Ok(self)
    }

    /// Wrapper of cuLinkAddFile
    #[allow(unused_unsafe)]
    unsafe fn add_file(mut self, input_type: CUjitInputType, path: &Path) -> Result<Self> {
        let filename = CString::new(path.to_str().unwrap()).expect("Invalid file path");
        let (nopts, mut opts, mut opt_vals) = self.cfg.pack();
        {
            contexted_call!(
                &self,
                cuLinkAddFile_v2,
                self.state,
                input_type,
                filename.as_ptr(),
                nopts,
                opts.as_mut_ptr(),
                opt_vals.as_mut_ptr()
            )?;
        }
        Ok(self)
    }

    /// Add a resouce into the linker stack.
    pub fn add(self, data: &Instruction) -> Result<Self> {
        Ok(match *data {
            Instruction::PTX(ref ptx) => unsafe {
                let cstr = CString::new(ptx.as_bytes()).expect("Invalid PTX String");
                self.add_data(data.input_type(), cstr.as_bytes_with_nul())?
            },
            Instruction::Cubin(ref bin) => unsafe { self.add_data(data.input_type(), &bin)? },
            Instruction::PTXFile(ref path) | Instruction::CubinFile(ref path) => unsafe {
                self.add_file(data.input_type(), path)?
            },
        })
    }

    /// Wrapper of cuLinkComplete
    ///
    /// LinkComplete returns a reference to cubin,
    /// which is managed by LinkState.
    /// Use owned strategy to avoid considering lifetime.
    pub fn complete(self) -> Result<Instruction> {
        let mut cb = null_mut();
        #[allow(unused_unsafe)]
        unsafe {
            contexted_call!(
                &self,
                cuLinkComplete,
                self.state,
                &mut cb as *mut _,
                null_mut()
            )?;
            Ok(Instruction::cubin(CStr::from_ptr(cb as _).to_bytes()))
        }
    }
}

/// Link PTX/cubin into a module
pub fn link<'ctx>(
    ctx: &'ctx Context,
    data: &[Instruction],
    opt: JITConfig,
) -> Result<Module<'ctx>> {
    let mut l = Linker::create(ctx, opt)?;
    for d in data {
        l = l.add(d)?;
    }
    let cubin = l.complete()?;
    Module::load(ctx, &cubin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _linker = Linker::create(&ctx, JITConfig::default())?;
        Ok(())
    }

    #[test]
    fn ptx_file() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let linker = Linker::create(&ctx, JITConfig::default())?;
        let data = Instruction::ptx_file(Path::new("tests/data/add.ptx"))?;
        linker.add(&data)?;
        Ok(())
    }

    #[test]
    fn linking() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();

        let data_add = Instruction::ptx_file(Path::new("tests/data/add.ptx"))?;
        let data_sub = Instruction::ptx_file(Path::new("tests/data/sub.ptx"))?;
        let _module = Linker::create(&ctx, JITConfig::default())?
            .add(&data_add)?
            .add(&data_sub)?
            .complete()?;
        Ok(())
    }

    #[ignore] // FIXME Causes CUDA_ERROR_NO_BINARY_FOR_GPU
    #[test]
    fn cubin_file() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let linker = Linker::create(&ctx, JITConfig::default())?;
        let data = Instruction::cubin_file(Path::new("tests/data/add.cubin"))?;
        linker.add(&data)?;
        Ok(())
    }
}
