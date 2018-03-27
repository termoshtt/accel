#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

pub const __CUDA_API_VERSION: ::std::os::raw::c_uint = 8000;
pub const CUDA_VERSION: ::std::os::raw::c_uint = 8000;
pub const CU_IPC_HANDLE_SIZE: ::std::os::raw::c_uint = 64;
pub const CU_MEMHOSTALLOC_PORTABLE: ::std::os::raw::c_uint = 1;
pub const CU_MEMHOSTALLOC_DEVICEMAP: ::std::os::raw::c_uint = 2;
pub const CU_MEMHOSTALLOC_WRITECOMBINED: ::std::os::raw::c_uint = 4;
pub const CU_MEMHOSTREGISTER_PORTABLE: ::std::os::raw::c_uint = 1;
pub const CU_MEMHOSTREGISTER_DEVICEMAP: ::std::os::raw::c_uint = 2;
pub const CU_MEMHOSTREGISTER_IOMEMORY: ::std::os::raw::c_uint = 4;
pub const CUDA_ARRAY3D_LAYERED: ::std::os::raw::c_uint = 1;
pub const CUDA_ARRAY3D_2DARRAY: ::std::os::raw::c_uint = 1;
pub const CUDA_ARRAY3D_SURFACE_LDST: ::std::os::raw::c_uint = 2;
pub const CUDA_ARRAY3D_CUBEMAP: ::std::os::raw::c_uint = 4;
pub const CUDA_ARRAY3D_TEXTURE_GATHER: ::std::os::raw::c_uint = 8;
pub const CUDA_ARRAY3D_DEPTH_TEXTURE: ::std::os::raw::c_uint = 16;
pub const CU_TRSA_OVERRIDE_FORMAT: ::std::os::raw::c_uint = 1;
pub const CU_TRSF_READ_AS_INTEGER: ::std::os::raw::c_uint = 1;
pub const CU_TRSF_NORMALIZED_COORDINATES: ::std::os::raw::c_uint = 2;
pub const CU_TRSF_SRGB: ::std::os::raw::c_uint = 16;
pub const CU_PARAM_TR_DEFAULT: ::std::os::raw::c_int = -1;

pub type cuuint32_t = u32;
pub type cuuint64_t = u64;
pub type CUdeviceptr = ::std::os::raw::c_ulonglong;
pub type CUdevice = ::std::os::raw::c_int;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUctx_st {
    pub _address: u8,
}
impl Clone for CUctx_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUcontext = *mut CUctx_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUmod_st {
    pub _address: u8,
}
impl Clone for CUmod_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUmodule = *mut CUmod_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUfunc_st {
    pub _address: u8,
}
impl Clone for CUfunc_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUfunction = *mut CUfunc_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUarray_st {
    pub _address: u8,
}
impl Clone for CUarray_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUarray = *mut CUarray_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUmipmappedArray_st {
    pub _address: u8,
}
impl Clone for CUmipmappedArray_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUmipmappedArray = *mut CUmipmappedArray_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUtexref_st {
    pub _address: u8,
}
impl Clone for CUtexref_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUtexref = *mut CUtexref_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUsurfref_st {
    pub _address: u8,
}
impl Clone for CUsurfref_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUsurfref = *mut CUsurfref_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUevent_st {
    pub _address: u8,
}
impl Clone for CUevent_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUevent = *mut CUevent_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUstream_st {
    pub _address: u8,
}
impl Clone for CUstream_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUstream = *mut CUstream_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUgraphicsResource_st {
    pub _address: u8,
}
impl Clone for CUgraphicsResource_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUgraphicsResource = *mut CUgraphicsResource_st;
pub type CUtexObject = ::std::os::raw::c_ulonglong;
pub type CUsurfObject = ::std::os::raw::c_ulonglong;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUuuid_st {
    pub bytes: [::std::os::raw::c_char; 16usize],
}
impl Clone for CUuuid_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUuuid = CUuuid_st;
#[repr(C)]
pub struct CUipcEventHandle_st {
    pub reserved: [::std::os::raw::c_char; 64usize],
}
pub type CUipcEventHandle = CUipcEventHandle_st;
#[repr(C)]
pub struct CUipcMemHandle_st {
    pub reserved: [::std::os::raw::c_char; 64usize],
}
pub type CUipcMemHandle = CUipcMemHandle_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUipcMem_flags_enum {
    CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1,
}
pub type CUipcMem_flags = CUipcMem_flags_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUmemAttach_flags_enum {
    CU_MEM_ATTACH_GLOBAL = 1,
    CU_MEM_ATTACH_HOST = 2,
    CU_MEM_ATTACH_SINGLE = 4,
}
pub type CUmemAttach_flags = CUmemAttach_flags_enum;
pub const CUctx_flags_enum_CU_CTX_BLOCKING_SYNC: CUctx_flags_enum = CUctx_flags_enum::CU_CTX_SCHED_BLOCKING_SYNC;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUctx_flags_enum {
    CU_CTX_SCHED_AUTO = 0,
    CU_CTX_SCHED_SPIN = 1,
    CU_CTX_SCHED_YIELD = 2,
    CU_CTX_SCHED_BLOCKING_SYNC = 4,
    CU_CTX_SCHED_MASK = 7,
    CU_CTX_MAP_HOST = 8,
    CU_CTX_LMEM_RESIZE_TO_MAX = 16,
    CU_CTX_FLAGS_MASK = 31,
}
pub type CUctx_flags = CUctx_flags_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUstream_flags_enum {
    CU_STREAM_DEFAULT = 0,
    CU_STREAM_NON_BLOCKING = 1,
}
pub type CUstream_flags = CUstream_flags_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUevent_flags_enum {
    CU_EVENT_DEFAULT = 0,
    CU_EVENT_BLOCKING_SYNC = 1,
    CU_EVENT_DISABLE_TIMING = 2,
    CU_EVENT_INTERPROCESS = 4,
}
pub type CUevent_flags = CUevent_flags_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUstreamWaitValue_flags_enum {
    CU_STREAM_WAIT_VALUE_GEQ = 0,
    CU_STREAM_WAIT_VALUE_EQ = 1,
    CU_STREAM_WAIT_VALUE_AND = 2,
    CU_STREAM_WAIT_VALUE_FLUSH = 1073741824,
}
pub type CUstreamWaitValue_flags = CUstreamWaitValue_flags_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUstreamWriteValue_flags_enum {
    CU_STREAM_WRITE_VALUE_DEFAULT = 0,
    CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 1,
}
pub type CUstreamWriteValue_flags = CUstreamWriteValue_flags_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUstreamBatchMemOpType_enum {
    CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,
    CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,
    CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3,
}
pub type CUstreamBatchMemOpType = CUstreamBatchMemOpType_enum;
#[repr(C)]
#[derive(Copy)]
pub union CUstreamBatchMemOpParams_union {
    pub operation: CUstreamBatchMemOpType,
    pub waitValue: CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st,
    pub writeValue: CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st,
    pub flushRemoteWrites: CUstreamBatchMemOpParams_union_CUstreamMemOpFlushRemoteWritesParams_st,
    pub pad: [cuuint64_t; 6usize],
}
#[repr(C)]
#[derive(Copy)]
pub struct CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st {
    pub operation: CUstreamBatchMemOpType,
    pub address: CUdeviceptr,
    pub __bindgen_anon_1: CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1,
    pub flags: ::std::os::raw::c_uint,
    pub alias: CUdeviceptr,
}
#[repr(C)]
#[derive(Copy)]
pub union CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1 {
    pub value: cuuint32_t,
    pub pad: cuuint64_t,
}
impl Clone for CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st {
    fn clone(&self) -> Self {
        *self
    }
}
#[repr(C)]
#[derive(Copy)]
pub struct CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st {
    pub operation: CUstreamBatchMemOpType,
    pub address: CUdeviceptr,
    pub __bindgen_anon_1: CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st__bindgen_ty_1,
    pub flags: ::std::os::raw::c_uint,
    pub alias: CUdeviceptr,
}
#[repr(C)]
#[derive(Copy)]
pub union CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st__bindgen_ty_1 {
    pub value: cuuint32_t,
    pub pad: cuuint64_t,
}
impl Clone for CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st__bindgen_ty_1 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st {
    fn clone(&self) -> Self {
        *self
    }
}
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUstreamBatchMemOpParams_union_CUstreamMemOpFlushRemoteWritesParams_st {
    pub operation: CUstreamBatchMemOpType,
    pub flags: ::std::os::raw::c_uint,
}
impl Clone for CUstreamBatchMemOpParams_union_CUstreamMemOpFlushRemoteWritesParams_st {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for CUstreamBatchMemOpParams_union {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUstreamBatchMemOpParams = CUstreamBatchMemOpParams_union;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUoccupancy_flags_enum {
    CU_OCCUPANCY_DEFAULT = 0,
    CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 1,
}
pub type CUoccupancy_flags = CUoccupancy_flags_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8 = 1,
    CU_AD_FORMAT_UNSIGNED_INT16 = 2,
    CU_AD_FORMAT_UNSIGNED_INT32 = 3,
    CU_AD_FORMAT_SIGNED_INT8 = 8,
    CU_AD_FORMAT_SIGNED_INT16 = 9,
    CU_AD_FORMAT_SIGNED_INT32 = 10,
    CU_AD_FORMAT_HALF = 16,
    CU_AD_FORMAT_FLOAT = 32,
}
pub type CUarray_format = CUarray_format_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUaddress_mode_enum {
    CU_TR_ADDRESS_MODE_WRAP = 0,
    CU_TR_ADDRESS_MODE_CLAMP = 1,
    CU_TR_ADDRESS_MODE_MIRROR = 2,
    CU_TR_ADDRESS_MODE_BORDER = 3,
}
pub type CUaddress_mode = CUaddress_mode_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUfilter_mode_enum {
    CU_TR_FILTER_MODE_POINT = 0,
    CU_TR_FILTER_MODE_LINEAR = 1,
}
pub type CUfilter_mode = CUfilter_mode_enum;
pub const CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK: CUdevice_attribute_enum =
    CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK;
pub const CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK: CUdevice_attribute_enum =
    CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK;
pub const CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH: CUdevice_attribute_enum =
    CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH;
pub const CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT: CUdevice_attribute_enum =
    CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT;
pub const CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES: CUdevice_attribute_enum =
    CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_MAX = 92,
}
pub type CUdevice_attribute = CUdevice_attribute_enum;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUdevprop_st {
    pub maxThreadsPerBlock: ::std::os::raw::c_int,
    pub maxThreadsDim: [::std::os::raw::c_int; 3usize],
    pub maxGridSize: [::std::os::raw::c_int; 3usize],
    pub sharedMemPerBlock: ::std::os::raw::c_int,
    pub totalConstantMemory: ::std::os::raw::c_int,
    pub SIMDWidth: ::std::os::raw::c_int,
    pub memPitch: ::std::os::raw::c_int,
    pub regsPerBlock: ::std::os::raw::c_int,
    pub clockRate: ::std::os::raw::c_int,
    pub textureAlign: ::std::os::raw::c_int,
}
impl Clone for CUdevprop_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUdevprop = CUdevprop_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
}
pub type CUpointer_attribute = CUpointer_attribute_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUfunction_attribute_enum {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    CU_FUNC_ATTRIBUTE_MAX = 8,
}
pub type CUfunction_attribute = CUfunction_attribute_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUfunc_cache_enum {
    CU_FUNC_CACHE_PREFER_NONE = 0,
    CU_FUNC_CACHE_PREFER_SHARED = 1,
    CU_FUNC_CACHE_PREFER_L1 = 2,
    CU_FUNC_CACHE_PREFER_EQUAL = 3,
}
pub type CUfunc_cache = CUfunc_cache_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUsharedconfig_enum {
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0,
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 1,
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 2,
}
pub type CUsharedconfig = CUsharedconfig_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST = 1,
    CU_MEMORYTYPE_DEVICE = 2,
    CU_MEMORYTYPE_ARRAY = 3,
    CU_MEMORYTYPE_UNIFIED = 4,
}
pub type CUmemorytype = CUmemorytype_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUcomputemode_enum {
    CU_COMPUTEMODE_DEFAULT = 0,
    CU_COMPUTEMODE_PROHIBITED = 2,
    CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3,
}
pub type CUcomputemode = CUcomputemode_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUmem_advise_enum {
    CU_MEM_ADVISE_SET_READ_MOSTLY = 1,
    CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2,
    CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3,
    CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4,
    CU_MEM_ADVISE_SET_ACCESSED_BY = 5,
    CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6,
}
pub type CUmem_advise = CUmem_advise_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUmem_range_attribute_enum {
    CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1,
    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2,
    CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3,
    CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4,
}
pub type CUmem_range_attribute = CUmem_range_attribute_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_NUM_OPTIONS = 17,
}
pub type CUjit_option = CUjit_option_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUjit_target_enum {
    CU_TARGET_COMPUTE_10 = 10,
    CU_TARGET_COMPUTE_11 = 11,
    CU_TARGET_COMPUTE_12 = 12,
    CU_TARGET_COMPUTE_13 = 13,
    CU_TARGET_COMPUTE_20 = 20,
    CU_TARGET_COMPUTE_21 = 21,
    CU_TARGET_COMPUTE_30 = 30,
    CU_TARGET_COMPUTE_32 = 32,
    CU_TARGET_COMPUTE_35 = 35,
    CU_TARGET_COMPUTE_37 = 37,
    CU_TARGET_COMPUTE_50 = 50,
    CU_TARGET_COMPUTE_52 = 52,
    CU_TARGET_COMPUTE_53 = 53,
    CU_TARGET_COMPUTE_60 = 60,
    CU_TARGET_COMPUTE_61 = 61,
    CU_TARGET_COMPUTE_62 = 62,
}
pub type CUjit_target = CUjit_target_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUjit_fallback_enum {
    CU_PREFER_PTX = 0,
    CU_PREFER_BINARY = 1,
}
pub type CUjit_fallback = CUjit_fallback_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUjit_cacheMode_enum {
    CU_JIT_CACHE_OPTION_NONE = 0,
    CU_JIT_CACHE_OPTION_CG = 1,
    CU_JIT_CACHE_OPTION_CA = 2,
}
pub type CUjit_cacheMode = CUjit_cacheMode_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUjitInputType_enum {
    CU_JIT_INPUT_CUBIN = 0,
    CU_JIT_INPUT_PTX = 1,
    CU_JIT_INPUT_FATBINARY = 2,
    CU_JIT_INPUT_OBJECT = 3,
    CU_JIT_INPUT_LIBRARY = 4,
    CU_JIT_NUM_INPUT_TYPES = 5,
}
pub type CUjitInputType = CUjitInputType_enum;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUlinkState_st {
    pub _address: u8,
}
impl Clone for CUlinkState_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUlinkState = *mut CUlinkState_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUgraphicsRegisterFlags_enum {
    CU_GRAPHICS_REGISTER_FLAGS_NONE = 0,
    CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 1,
    CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2,
    CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4,
    CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 8,
}
pub type CUgraphicsRegisterFlags = CUgraphicsRegisterFlags_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUgraphicsMapResourceFlags_enum {
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0,
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 1,
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 2,
}
pub type CUgraphicsMapResourceFlags = CUgraphicsMapResourceFlags_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUarray_cubemap_face_enum {
    CU_CUBEMAP_FACE_POSITIVE_X = 0,
    CU_CUBEMAP_FACE_NEGATIVE_X = 1,
    CU_CUBEMAP_FACE_POSITIVE_Y = 2,
    CU_CUBEMAP_FACE_NEGATIVE_Y = 3,
    CU_CUBEMAP_FACE_POSITIVE_Z = 4,
    CU_CUBEMAP_FACE_NEGATIVE_Z = 5,
}
pub type CUarray_cubemap_face = CUarray_cubemap_face_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUlimit_enum {
    CU_LIMIT_STACK_SIZE = 0,
    CU_LIMIT_PRINTF_FIFO_SIZE = 1,
    CU_LIMIT_MALLOC_HEAP_SIZE = 2,
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3,
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4,
    CU_LIMIT_MAX = 5,
}
pub type CUlimit = CUlimit_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUresourcetype_enum {
    CU_RESOURCE_TYPE_ARRAY = 0,
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1,
    CU_RESOURCE_TYPE_LINEAR = 2,
    CU_RESOURCE_TYPE_PITCH2D = 3,
}
pub type CUresourcetype = CUresourcetype_enum;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cudaError_t {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_UNKNOWN = 999,
}
pub type CUresult = cudaError_t;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUdevice_P2PAttribute_enum {
    CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 1,
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 2,
    CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 3,
}
pub type CUdevice_P2PAttribute = CUdevice_P2PAttribute_enum;
pub type CUstreamCallback = ::std::option::Option<
    unsafe extern "C" fn(hStream: CUstream, status: CUresult, userData: *mut ::std::os::raw::c_void),
>;
pub type CUoccupancyB2DSize =
    ::std::option::Option<unsafe extern "C" fn(blockSize: ::std::os::raw::c_int) -> ::std::os::raw::c_ulong>;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUDA_MEMCPY2D_st {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const ::std::os::raw::c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub srcPitch: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut ::std::os::raw::c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub dstPitch: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
}
impl Clone for CUDA_MEMCPY2D_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUDA_MEMCPY2D = CUDA_MEMCPY2D_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUDA_MEMCPY3D_st {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcZ: usize,
    pub srcLOD: usize,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const ::std::os::raw::c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub reserved0: *mut ::std::os::raw::c_void,
    pub srcPitch: usize,
    pub srcHeight: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstZ: usize,
    pub dstLOD: usize,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut ::std::os::raw::c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub reserved1: *mut ::std::os::raw::c_void,
    pub dstPitch: usize,
    pub dstHeight: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
    pub Depth: usize,
}
impl Clone for CUDA_MEMCPY3D_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUDA_MEMCPY3D = CUDA_MEMCPY3D_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUDA_MEMCPY3D_PEER_st {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcZ: usize,
    pub srcLOD: usize,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const ::std::os::raw::c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub srcContext: CUcontext,
    pub srcPitch: usize,
    pub srcHeight: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstZ: usize,
    pub dstLOD: usize,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut ::std::os::raw::c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub dstContext: CUcontext,
    pub dstPitch: usize,
    pub dstHeight: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
    pub Depth: usize,
}
impl Clone for CUDA_MEMCPY3D_PEER_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUDA_MEMCPY3D_PEER = CUDA_MEMCPY3D_PEER_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUDA_ARRAY_DESCRIPTOR_st {
    pub Width: usize,
    pub Height: usize,
    pub Format: CUarray_format,
    pub NumChannels: ::std::os::raw::c_uint,
}
impl Clone for CUDA_ARRAY_DESCRIPTOR_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUDA_ARRAY_DESCRIPTOR = CUDA_ARRAY_DESCRIPTOR_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUDA_ARRAY3D_DESCRIPTOR_st {
    pub Width: usize,
    pub Height: usize,
    pub Depth: usize,
    pub Format: CUarray_format,
    pub NumChannels: ::std::os::raw::c_uint,
    pub Flags: ::std::os::raw::c_uint,
}
impl Clone for CUDA_ARRAY3D_DESCRIPTOR_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUDA_ARRAY3D_DESCRIPTOR = CUDA_ARRAY3D_DESCRIPTOR_st;
#[repr(C)]
#[derive(Copy)]
pub struct CUDA_RESOURCE_DESC_st {
    pub resType: CUresourcetype,
    pub res: CUDA_RESOURCE_DESC_st__bindgen_ty_1,
    pub flags: ::std::os::raw::c_uint,
}
#[repr(C)]
#[derive(Copy)]
pub union CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
    pub array: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1,
    pub mipmap: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2,
    pub linear: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3,
    pub pitch2D: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4,
    pub reserved: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5,
}
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
    pub hArray: CUarray,
}
impl Clone for CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
    fn clone(&self) -> Self {
        *self
    }
}
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2 {
    pub hMipmappedArray: CUmipmappedArray,
}
impl Clone for CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2 {
    fn clone(&self) -> Self {
        *self
    }
}
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3 {
    pub devPtr: CUdeviceptr,
    pub format: CUarray_format,
    pub numChannels: ::std::os::raw::c_uint,
    pub sizeInBytes: usize,
}
impl Clone for CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3 {
    fn clone(&self) -> Self {
        *self
    }
}
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4 {
    pub devPtr: CUdeviceptr,
    pub format: CUarray_format,
    pub numChannels: ::std::os::raw::c_uint,
    pub width: usize,
    pub height: usize,
    pub pitchInBytes: usize,
}
impl Clone for CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4 {
    fn clone(&self) -> Self {
        *self
    }
}
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5 {
    pub reserved: [::std::os::raw::c_int; 32usize],
}
impl Clone for CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for CUDA_RESOURCE_DESC_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUDA_RESOURCE_DESC = CUDA_RESOURCE_DESC_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUDA_TEXTURE_DESC_st {
    pub addressMode: [CUaddress_mode; 3usize],
    pub filterMode: CUfilter_mode,
    pub flags: ::std::os::raw::c_uint,
    pub maxAnisotropy: ::std::os::raw::c_uint,
    pub mipmapFilterMode: CUfilter_mode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub borderColor: [f32; 4usize],
    pub reserved: [::std::os::raw::c_int; 12usize],
}
impl Clone for CUDA_TEXTURE_DESC_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUDA_TEXTURE_DESC = CUDA_TEXTURE_DESC_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUresourceViewFormat_enum {
    CU_RES_VIEW_FORMAT_NONE = 0,
    CU_RES_VIEW_FORMAT_UINT_1X8 = 1,
    CU_RES_VIEW_FORMAT_UINT_2X8 = 2,
    CU_RES_VIEW_FORMAT_UINT_4X8 = 3,
    CU_RES_VIEW_FORMAT_SINT_1X8 = 4,
    CU_RES_VIEW_FORMAT_SINT_2X8 = 5,
    CU_RES_VIEW_FORMAT_SINT_4X8 = 6,
    CU_RES_VIEW_FORMAT_UINT_1X16 = 7,
    CU_RES_VIEW_FORMAT_UINT_2X16 = 8,
    CU_RES_VIEW_FORMAT_UINT_4X16 = 9,
    CU_RES_VIEW_FORMAT_SINT_1X16 = 10,
    CU_RES_VIEW_FORMAT_SINT_2X16 = 11,
    CU_RES_VIEW_FORMAT_SINT_4X16 = 12,
    CU_RES_VIEW_FORMAT_UINT_1X32 = 13,
    CU_RES_VIEW_FORMAT_UINT_2X32 = 14,
    CU_RES_VIEW_FORMAT_UINT_4X32 = 15,
    CU_RES_VIEW_FORMAT_SINT_1X32 = 16,
    CU_RES_VIEW_FORMAT_SINT_2X32 = 17,
    CU_RES_VIEW_FORMAT_SINT_4X32 = 18,
    CU_RES_VIEW_FORMAT_FLOAT_1X16 = 19,
    CU_RES_VIEW_FORMAT_FLOAT_2X16 = 20,
    CU_RES_VIEW_FORMAT_FLOAT_4X16 = 21,
    CU_RES_VIEW_FORMAT_FLOAT_1X32 = 22,
    CU_RES_VIEW_FORMAT_FLOAT_2X32 = 23,
    CU_RES_VIEW_FORMAT_FLOAT_4X32 = 24,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28,
    CU_RES_VIEW_FORMAT_SIGNED_BC4 = 29,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30,
    CU_RES_VIEW_FORMAT_SIGNED_BC5 = 31,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32,
    CU_RES_VIEW_FORMAT_SIGNED_BC6H = 33,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34,
}
pub type CUresourceViewFormat = CUresourceViewFormat_enum;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUDA_RESOURCE_VIEW_DESC_st {
    pub format: CUresourceViewFormat,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub firstMipmapLevel: ::std::os::raw::c_uint,
    pub lastMipmapLevel: ::std::os::raw::c_uint,
    pub firstLayer: ::std::os::raw::c_uint,
    pub lastLayer: ::std::os::raw::c_uint,
    pub reserved: [::std::os::raw::c_uint; 16usize],
}
impl Clone for CUDA_RESOURCE_VIEW_DESC_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUDA_RESOURCE_VIEW_DESC = CUDA_RESOURCE_VIEW_DESC_st;
#[repr(C)]
#[derive(Debug, Copy)]
pub struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st {
    pub p2pToken: ::std::os::raw::c_ulonglong,
    pub vaSpaceToken: ::std::os::raw::c_uint,
}
impl Clone for CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st {
    fn clone(&self) -> Self {
        *self
    }
}
pub type CUDA_POINTER_ATTRIBUTE_P2P_TOKENS = CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st;
extern "C" {
    pub fn cuGetErrorString(error: CUresult, pStr: *mut *const ::std::os::raw::c_char) -> CUresult;

    pub fn cuGetErrorName(error: CUresult, pStr: *mut *const ::std::os::raw::c_char) -> CUresult;

    pub fn cuInit(Flags: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuDriverGetVersion(driverVersion: *mut ::std::os::raw::c_int) -> CUresult;

    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: ::std::os::raw::c_int) -> CUresult;

    pub fn cuDeviceGetCount(count: *mut ::std::os::raw::c_int) -> CUresult;

    pub fn cuDeviceGetName(name: *mut ::std::os::raw::c_char, len: ::std::os::raw::c_int, dev: CUdevice) -> CUresult;

    pub fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;

    pub fn cuDeviceGetAttribute(pi: *mut ::std::os::raw::c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult;

    pub fn cuDeviceGetProperties(prop: *mut CUdevprop, dev: CUdevice) -> CUresult;

    pub fn cuDeviceComputeCapability(
        major: *mut ::std::os::raw::c_int,
        minor: *mut ::std::os::raw::c_int,
        dev: CUdevice,
    ) -> CUresult;

    pub fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;

    pub fn cuDevicePrimaryCtxRelease(dev: CUdevice) -> CUresult;

    pub fn cuDevicePrimaryCtxSetFlags(dev: CUdevice, flags: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuDevicePrimaryCtxGetState(
        dev: CUdevice,
        flags: *mut ::std::os::raw::c_uint,
        active: *mut ::std::os::raw::c_int,
    ) -> CUresult;

    pub fn cuDevicePrimaryCtxReset(dev: CUdevice) -> CUresult;

    pub fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: ::std::os::raw::c_uint, dev: CUdevice) -> CUresult;

    pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;

    pub fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult;

    pub fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult;

    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;

    pub fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;

    pub fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult;

    pub fn cuCtxGetFlags(flags: *mut ::std::os::raw::c_uint) -> CUresult;

    pub fn cuCtxSynchronize() -> CUresult;

    pub fn cuCtxSetLimit(limit: CUlimit, value: usize) -> CUresult;

    pub fn cuCtxGetLimit(pvalue: *mut usize, limit: CUlimit) -> CUresult;

    pub fn cuCtxGetCacheConfig(pconfig: *mut CUfunc_cache) -> CUresult;

    pub fn cuCtxSetCacheConfig(config: CUfunc_cache) -> CUresult;

    pub fn cuCtxGetSharedMemConfig(pConfig: *mut CUsharedconfig) -> CUresult;

    pub fn cuCtxSetSharedMemConfig(config: CUsharedconfig) -> CUresult;

    pub fn cuCtxGetApiVersion(ctx: CUcontext, version: *mut ::std::os::raw::c_uint) -> CUresult;

    pub fn cuCtxGetStreamPriorityRange(
        leastPriority: *mut ::std::os::raw::c_int,
        greatestPriority: *mut ::std::os::raw::c_int,
    ) -> CUresult;

    pub fn cuCtxAttach(pctx: *mut CUcontext, flags: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuCtxDetach(ctx: CUcontext) -> CUresult;

    pub fn cuModuleLoad(module: *mut CUmodule, fname: *const ::std::os::raw::c_char) -> CUresult;

    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const ::std::os::raw::c_void) -> CUresult;

    pub fn cuModuleLoadDataEx(
        module: *mut CUmodule,
        image: *const ::std::os::raw::c_void,
        numOptions: ::std::os::raw::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::std::os::raw::c_void,
    ) -> CUresult;

    pub fn cuModuleLoadFatBinary(module: *mut CUmodule, fatCubin: *const ::std::os::raw::c_void) -> CUresult;

    pub fn cuModuleUnload(hmod: CUmodule) -> CUresult;

    pub fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const ::std::os::raw::c_char)
        -> CUresult;

    pub fn cuModuleGetGlobal_v2(
        dptr: *mut CUdeviceptr,
        bytes: *mut usize,
        hmod: CUmodule,
        name: *const ::std::os::raw::c_char,
    ) -> CUresult;

    pub fn cuModuleGetTexRef(pTexRef: *mut CUtexref, hmod: CUmodule, name: *const ::std::os::raw::c_char) -> CUresult;

    pub fn cuModuleGetSurfRef(
        pSurfRef: *mut CUsurfref,
        hmod: CUmodule,
        name: *const ::std::os::raw::c_char,
    ) -> CUresult;

    pub fn cuLinkCreate_v2(
        numOptions: ::std::os::raw::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::std::os::raw::c_void,
        stateOut: *mut CUlinkState,
    ) -> CUresult;

    pub fn cuLinkAddData_v2(
        state: CUlinkState,
        type_: CUjitInputType,
        data: *mut ::std::os::raw::c_void,
        size: usize,
        name: *const ::std::os::raw::c_char,
        numOptions: ::std::os::raw::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::std::os::raw::c_void,
    ) -> CUresult;

    pub fn cuLinkAddFile_v2(
        state: CUlinkState,
        type_: CUjitInputType,
        path: *const ::std::os::raw::c_char,
        numOptions: ::std::os::raw::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::std::os::raw::c_void,
    ) -> CUresult;

    pub fn cuLinkComplete(
        state: CUlinkState,
        cubinOut: *mut *mut ::std::os::raw::c_void,
        sizeOut: *mut usize,
    ) -> CUresult;

    pub fn cuLinkDestroy(state: CUlinkState) -> CUresult;

    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;

    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;

    pub fn cuMemAllocPitch_v2(
        dptr: *mut CUdeviceptr,
        pPitch: *mut usize,
        WidthInBytes: usize,
        Height: usize,
        ElementSizeBytes: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;

    pub fn cuMemGetAddressRange_v2(pbase: *mut CUdeviceptr, psize: *mut usize, dptr: CUdeviceptr) -> CUresult;

    pub fn cuMemAllocHost_v2(pp: *mut *mut ::std::os::raw::c_void, bytesize: usize) -> CUresult;

    pub fn cuMemFreeHost(p: *mut ::std::os::raw::c_void) -> CUresult;

    pub fn cuMemHostAlloc(
        pp: *mut *mut ::std::os::raw::c_void,
        bytesize: usize,
        Flags: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuMemHostGetDevicePointer_v2(
        pdptr: *mut CUdeviceptr,
        p: *mut ::std::os::raw::c_void,
        Flags: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuMemHostGetFlags(pFlags: *mut ::std::os::raw::c_uint, p: *mut ::std::os::raw::c_void) -> CUresult;

    pub fn cuMemAllocManaged(dptr: *mut CUdeviceptr, bytesize: usize, flags: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuDeviceGetByPCIBusId(dev: *mut CUdevice, pciBusId: *const ::std::os::raw::c_char) -> CUresult;

    pub fn cuDeviceGetPCIBusId(
        pciBusId: *mut ::std::os::raw::c_char,
        len: ::std::os::raw::c_int,
        dev: CUdevice,
    ) -> CUresult;

    pub fn cuIpcGetEventHandle(pHandle: *mut CUipcEventHandle, event: CUevent) -> CUresult;

    pub fn cuIpcOpenEventHandle(phEvent: *mut CUevent, handle: CUipcEventHandle) -> CUresult;

    pub fn cuIpcGetMemHandle(pHandle: *mut CUipcMemHandle, dptr: CUdeviceptr) -> CUresult;

    pub fn cuIpcOpenMemHandle(
        pdptr: *mut CUdeviceptr,
        handle: CUipcMemHandle,
        Flags: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuIpcCloseMemHandle(dptr: CUdeviceptr) -> CUresult;

    pub fn cuMemHostRegister_v2(
        p: *mut ::std::os::raw::c_void,
        bytesize: usize,
        Flags: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuMemHostUnregister(p: *mut ::std::os::raw::c_void) -> CUresult;

    pub fn cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: usize) -> CUresult;

    pub fn cuMemcpyPeer(
        dstDevice: CUdeviceptr,
        dstContext: CUcontext,
        srcDevice: CUdeviceptr,
        srcContext: CUcontext,
        ByteCount: usize,
    ) -> CUresult;

    pub fn cuMemcpyHtoD_v2(
        dstDevice: CUdeviceptr,
        srcHost: *const ::std::os::raw::c_void,
        ByteCount: usize,
    ) -> CUresult;

    pub fn cuMemcpyDtoH_v2(dstHost: *mut ::std::os::raw::c_void, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult;

    pub fn cuMemcpyDtoD_v2(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult;

    pub fn cuMemcpyDtoA_v2(dstArray: CUarray, dstOffset: usize, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult;

    pub fn cuMemcpyAtoD_v2(dstDevice: CUdeviceptr, srcArray: CUarray, srcOffset: usize, ByteCount: usize) -> CUresult;

    pub fn cuMemcpyHtoA_v2(
        dstArray: CUarray,
        dstOffset: usize,
        srcHost: *const ::std::os::raw::c_void,
        ByteCount: usize,
    ) -> CUresult;

    pub fn cuMemcpyAtoH_v2(
        dstHost: *mut ::std::os::raw::c_void,
        srcArray: CUarray,
        srcOffset: usize,
        ByteCount: usize,
    ) -> CUresult;

    pub fn cuMemcpyAtoA_v2(
        dstArray: CUarray,
        dstOffset: usize,
        srcArray: CUarray,
        srcOffset: usize,
        ByteCount: usize,
    ) -> CUresult;

    pub fn cuMemcpy2D_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult;

    pub fn cuMemcpy2DUnaligned_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult;

    pub fn cuMemcpy3D_v2(pCopy: *const CUDA_MEMCPY3D) -> CUresult;

    pub fn cuMemcpy3DPeer(pCopy: *const CUDA_MEMCPY3D_PEER) -> CUresult;

    pub fn cuMemcpyAsync(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: usize, hStream: CUstream) -> CUresult;

    pub fn cuMemcpyPeerAsync(
        dstDevice: CUdeviceptr,
        dstContext: CUcontext,
        srcDevice: CUdeviceptr,
        srcContext: CUcontext,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuMemcpyHtoDAsync_v2(
        dstDevice: CUdeviceptr,
        srcHost: *const ::std::os::raw::c_void,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuMemcpyDtoHAsync_v2(
        dstHost: *mut ::std::os::raw::c_void,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuMemcpyDtoDAsync_v2(
        dstDevice: CUdeviceptr,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuMemcpyHtoAAsync_v2(
        dstArray: CUarray,
        dstOffset: usize,
        srcHost: *const ::std::os::raw::c_void,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuMemcpyAtoHAsync_v2(
        dstHost: *mut ::std::os::raw::c_void,
        srcArray: CUarray,
        srcOffset: usize,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuMemcpy2DAsync_v2(pCopy: *const CUDA_MEMCPY2D, hStream: CUstream) -> CUresult;

    pub fn cuMemcpy3DAsync_v2(pCopy: *const CUDA_MEMCPY3D, hStream: CUstream) -> CUresult;

    pub fn cuMemcpy3DPeerAsync(pCopy: *const CUDA_MEMCPY3D_PEER, hStream: CUstream) -> CUresult;

    pub fn cuMemsetD8_v2(dstDevice: CUdeviceptr, uc: ::std::os::raw::c_uchar, N: usize) -> CUresult;

    pub fn cuMemsetD16_v2(dstDevice: CUdeviceptr, us: ::std::os::raw::c_ushort, N: usize) -> CUresult;

    pub fn cuMemsetD32_v2(dstDevice: CUdeviceptr, ui: ::std::os::raw::c_uint, N: usize) -> CUresult;

    pub fn cuMemsetD2D8_v2(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        uc: ::std::os::raw::c_uchar,
        Width: usize,
        Height: usize,
    ) -> CUresult;

    pub fn cuMemsetD2D16_v2(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        us: ::std::os::raw::c_ushort,
        Width: usize,
        Height: usize,
    ) -> CUresult;

    pub fn cuMemsetD2D32_v2(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        ui: ::std::os::raw::c_uint,
        Width: usize,
        Height: usize,
    ) -> CUresult;

    pub fn cuMemsetD8Async(
        dstDevice: CUdeviceptr,
        uc: ::std::os::raw::c_uchar,
        N: usize,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuMemsetD16Async(
        dstDevice: CUdeviceptr,
        us: ::std::os::raw::c_ushort,
        N: usize,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuMemsetD32Async(
        dstDevice: CUdeviceptr,
        ui: ::std::os::raw::c_uint,
        N: usize,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuMemsetD2D8Async(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        uc: ::std::os::raw::c_uchar,
        Width: usize,
        Height: usize,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuMemsetD2D16Async(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        us: ::std::os::raw::c_ushort,
        Width: usize,
        Height: usize,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuMemsetD2D32Async(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        ui: ::std::os::raw::c_uint,
        Width: usize,
        Height: usize,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuArrayCreate_v2(pHandle: *mut CUarray, pAllocateArray: *const CUDA_ARRAY_DESCRIPTOR) -> CUresult;

    pub fn cuArrayGetDescriptor_v2(pArrayDescriptor: *mut CUDA_ARRAY_DESCRIPTOR, hArray: CUarray) -> CUresult;

    pub fn cuArrayDestroy(hArray: CUarray) -> CUresult;

    pub fn cuArray3DCreate_v2(pHandle: *mut CUarray, pAllocateArray: *const CUDA_ARRAY3D_DESCRIPTOR) -> CUresult;

    pub fn cuArray3DGetDescriptor_v2(pArrayDescriptor: *mut CUDA_ARRAY3D_DESCRIPTOR, hArray: CUarray) -> CUresult;

    pub fn cuMipmappedArrayCreate(
        pHandle: *mut CUmipmappedArray,
        pMipmappedArrayDesc: *const CUDA_ARRAY3D_DESCRIPTOR,
        numMipmapLevels: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuMipmappedArrayGetLevel(
        pLevelArray: *mut CUarray,
        hMipmappedArray: CUmipmappedArray,
        level: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuMipmappedArrayDestroy(hMipmappedArray: CUmipmappedArray) -> CUresult;

    pub fn cuPointerGetAttribute(
        data: *mut ::std::os::raw::c_void,
        attribute: CUpointer_attribute,
        ptr: CUdeviceptr,
    ) -> CUresult;

    pub fn cuMemPrefetchAsync(devPtr: CUdeviceptr, count: usize, dstDevice: CUdevice, hStream: CUstream) -> CUresult;

    pub fn cuMemAdvise(devPtr: CUdeviceptr, count: usize, advice: CUmem_advise, device: CUdevice) -> CUresult;

    pub fn cuMemRangeGetAttribute(
        data: *mut ::std::os::raw::c_void,
        dataSize: usize,
        attribute: CUmem_range_attribute,
        devPtr: CUdeviceptr,
        count: usize,
    ) -> CUresult;

    pub fn cuMemRangeGetAttributes(
        data: *mut *mut ::std::os::raw::c_void,
        dataSizes: *mut usize,
        attributes: *mut CUmem_range_attribute,
        numAttributes: usize,
        devPtr: CUdeviceptr,
        count: usize,
    ) -> CUresult;

    pub fn cuPointerSetAttribute(
        value: *const ::std::os::raw::c_void,
        attribute: CUpointer_attribute,
        ptr: CUdeviceptr,
    ) -> CUresult;

    pub fn cuPointerGetAttributes(
        numAttributes: ::std::os::raw::c_uint,
        attributes: *mut CUpointer_attribute,
        data: *mut *mut ::std::os::raw::c_void,
        ptr: CUdeviceptr,
    ) -> CUresult;

    pub fn cuStreamCreate(phStream: *mut CUstream, Flags: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuStreamCreateWithPriority(
        phStream: *mut CUstream,
        flags: ::std::os::raw::c_uint,
        priority: ::std::os::raw::c_int,
    ) -> CUresult;

    pub fn cuStreamGetPriority(hStream: CUstream, priority: *mut ::std::os::raw::c_int) -> CUresult;

    pub fn cuStreamGetFlags(hStream: CUstream, flags: *mut ::std::os::raw::c_uint) -> CUresult;

    pub fn cuStreamWaitEvent(hStream: CUstream, hEvent: CUevent, Flags: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuStreamAddCallback(
        hStream: CUstream,
        callback: CUstreamCallback,
        userData: *mut ::std::os::raw::c_void,
        flags: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuStreamAttachMemAsync(
        hStream: CUstream,
        dptr: CUdeviceptr,
        length: usize,
        flags: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuStreamQuery(hStream: CUstream) -> CUresult;

    pub fn cuStreamSynchronize(hStream: CUstream) -> CUresult;

    pub fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult;

    pub fn cuEventCreate(phEvent: *mut CUevent, Flags: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult;

    pub fn cuEventQuery(hEvent: CUevent) -> CUresult;

    pub fn cuEventSynchronize(hEvent: CUevent) -> CUresult;

    pub fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult;

    pub fn cuEventElapsedTime(pMilliseconds: *mut f32, hStart: CUevent, hEnd: CUevent) -> CUresult;

    pub fn cuStreamWaitValue32(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint32_t,
        flags: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuStreamWriteValue32(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint32_t,
        flags: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuStreamBatchMemOp(
        stream: CUstream,
        count: ::std::os::raw::c_uint,
        paramArray: *mut CUstreamBatchMemOpParams,
        flags: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuFuncGetAttribute(
        pi: *mut ::std::os::raw::c_int,
        attrib: CUfunction_attribute,
        hfunc: CUfunction,
    ) -> CUresult;

    pub fn cuFuncSetCacheConfig(hfunc: CUfunction, config: CUfunc_cache) -> CUresult;

    pub fn cuFuncSetSharedMemConfig(hfunc: CUfunction, config: CUsharedconfig) -> CUresult;

    pub fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: ::std::os::raw::c_uint,
        gridDimY: ::std::os::raw::c_uint,
        gridDimZ: ::std::os::raw::c_uint,
        blockDimX: ::std::os::raw::c_uint,
        blockDimY: ::std::os::raw::c_uint,
        blockDimZ: ::std::os::raw::c_uint,
        sharedMemBytes: ::std::os::raw::c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut ::std::os::raw::c_void,
        extra: *mut *mut ::std::os::raw::c_void,
    ) -> CUresult;

    pub fn cuFuncSetBlockShape(
        hfunc: CUfunction,
        x: ::std::os::raw::c_int,
        y: ::std::os::raw::c_int,
        z: ::std::os::raw::c_int,
    ) -> CUresult;

    pub fn cuFuncSetSharedSize(hfunc: CUfunction, bytes: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuParamSetSize(hfunc: CUfunction, numbytes: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuParamSeti(hfunc: CUfunction, offset: ::std::os::raw::c_int, value: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuParamSetf(hfunc: CUfunction, offset: ::std::os::raw::c_int, value: f32) -> CUresult;

    pub fn cuParamSetv(
        hfunc: CUfunction,
        offset: ::std::os::raw::c_int,
        ptr: *mut ::std::os::raw::c_void,
        numbytes: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuLaunch(f: CUfunction) -> CUresult;

    pub fn cuLaunchGrid(
        f: CUfunction,
        grid_width: ::std::os::raw::c_int,
        grid_height: ::std::os::raw::c_int,
    ) -> CUresult;

    pub fn cuLaunchGridAsync(
        f: CUfunction,
        grid_width: ::std::os::raw::c_int,
        grid_height: ::std::os::raw::c_int,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuParamSetTexRef(hfunc: CUfunction, texunit: ::std::os::raw::c_int, hTexRef: CUtexref) -> CUresult;

    pub fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks: *mut ::std::os::raw::c_int,
        func: CUfunction,
        blockSize: ::std::os::raw::c_int,
        dynamicSMemSize: usize,
    ) -> CUresult;

    pub fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks: *mut ::std::os::raw::c_int,
        func: CUfunction,
        blockSize: ::std::os::raw::c_int,
        dynamicSMemSize: usize,
        flags: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuOccupancyMaxPotentialBlockSize(
        minGridSize: *mut ::std::os::raw::c_int,
        blockSize: *mut ::std::os::raw::c_int,
        func: CUfunction,
        blockSizeToDynamicSMemSize: CUoccupancyB2DSize,
        dynamicSMemSize: usize,
        blockSizeLimit: ::std::os::raw::c_int,
    ) -> CUresult;

    pub fn cuOccupancyMaxPotentialBlockSizeWithFlags(
        minGridSize: *mut ::std::os::raw::c_int,
        blockSize: *mut ::std::os::raw::c_int,
        func: CUfunction,
        blockSizeToDynamicSMemSize: CUoccupancyB2DSize,
        dynamicSMemSize: usize,
        blockSizeLimit: ::std::os::raw::c_int,
        flags: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuTexRefSetArray(hTexRef: CUtexref, hArray: CUarray, Flags: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuTexRefSetMipmappedArray(
        hTexRef: CUtexref,
        hMipmappedArray: CUmipmappedArray,
        Flags: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuTexRefSetAddress_v2(
        ByteOffset: *mut usize,
        hTexRef: CUtexref,
        dptr: CUdeviceptr,
        bytes: usize,
    ) -> CUresult;

    pub fn cuTexRefSetAddress2D_v3(
        hTexRef: CUtexref,
        desc: *const CUDA_ARRAY_DESCRIPTOR,
        dptr: CUdeviceptr,
        Pitch: usize,
    ) -> CUresult;

    pub fn cuTexRefSetFormat(
        hTexRef: CUtexref,
        fmt: CUarray_format,
        NumPackedComponents: ::std::os::raw::c_int,
    ) -> CUresult;

    pub fn cuTexRefSetAddressMode(hTexRef: CUtexref, dim: ::std::os::raw::c_int, am: CUaddress_mode) -> CUresult;

    pub fn cuTexRefSetFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) -> CUresult;

    pub fn cuTexRefSetMipmapFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) -> CUresult;

    pub fn cuTexRefSetMipmapLevelBias(hTexRef: CUtexref, bias: f32) -> CUresult;

    pub fn cuTexRefSetMipmapLevelClamp(
        hTexRef: CUtexref,
        minMipmapLevelClamp: f32,
        maxMipmapLevelClamp: f32,
    ) -> CUresult;

    pub fn cuTexRefSetMaxAnisotropy(hTexRef: CUtexref, maxAniso: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuTexRefSetBorderColor(hTexRef: CUtexref, pBorderColor: *mut f32) -> CUresult;

    pub fn cuTexRefSetFlags(hTexRef: CUtexref, Flags: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuTexRefGetAddress_v2(pdptr: *mut CUdeviceptr, hTexRef: CUtexref) -> CUresult;

    pub fn cuTexRefGetArray(phArray: *mut CUarray, hTexRef: CUtexref) -> CUresult;

    pub fn cuTexRefGetMipmappedArray(phMipmappedArray: *mut CUmipmappedArray, hTexRef: CUtexref) -> CUresult;

    pub fn cuTexRefGetAddressMode(pam: *mut CUaddress_mode, hTexRef: CUtexref, dim: ::std::os::raw::c_int) -> CUresult;

    pub fn cuTexRefGetFilterMode(pfm: *mut CUfilter_mode, hTexRef: CUtexref) -> CUresult;

    pub fn cuTexRefGetFormat(
        pFormat: *mut CUarray_format,
        pNumChannels: *mut ::std::os::raw::c_int,
        hTexRef: CUtexref,
    ) -> CUresult;

    pub fn cuTexRefGetMipmapFilterMode(pfm: *mut CUfilter_mode, hTexRef: CUtexref) -> CUresult;

    pub fn cuTexRefGetMipmapLevelBias(pbias: *mut f32, hTexRef: CUtexref) -> CUresult;

    pub fn cuTexRefGetMipmapLevelClamp(
        pminMipmapLevelClamp: *mut f32,
        pmaxMipmapLevelClamp: *mut f32,
        hTexRef: CUtexref,
    ) -> CUresult;

    pub fn cuTexRefGetMaxAnisotropy(pmaxAniso: *mut ::std::os::raw::c_int, hTexRef: CUtexref) -> CUresult;

    pub fn cuTexRefGetBorderColor(pBorderColor: *mut f32, hTexRef: CUtexref) -> CUresult;

    pub fn cuTexRefGetFlags(pFlags: *mut ::std::os::raw::c_uint, hTexRef: CUtexref) -> CUresult;

    pub fn cuTexRefCreate(pTexRef: *mut CUtexref) -> CUresult;

    pub fn cuTexRefDestroy(hTexRef: CUtexref) -> CUresult;

    pub fn cuSurfRefSetArray(hSurfRef: CUsurfref, hArray: CUarray, Flags: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuSurfRefGetArray(phArray: *mut CUarray, hSurfRef: CUsurfref) -> CUresult;

    pub fn cuTexObjectCreate(
        pTexObject: *mut CUtexObject,
        pResDesc: *const CUDA_RESOURCE_DESC,
        pTexDesc: *const CUDA_TEXTURE_DESC,
        pResViewDesc: *const CUDA_RESOURCE_VIEW_DESC,
    ) -> CUresult;

    pub fn cuTexObjectDestroy(texObject: CUtexObject) -> CUresult;

    pub fn cuTexObjectGetResourceDesc(pResDesc: *mut CUDA_RESOURCE_DESC, texObject: CUtexObject) -> CUresult;

    pub fn cuTexObjectGetTextureDesc(pTexDesc: *mut CUDA_TEXTURE_DESC, texObject: CUtexObject) -> CUresult;

    pub fn cuTexObjectGetResourceViewDesc(
        pResViewDesc: *mut CUDA_RESOURCE_VIEW_DESC,
        texObject: CUtexObject,
    ) -> CUresult;

    pub fn cuSurfObjectCreate(pSurfObject: *mut CUsurfObject, pResDesc: *const CUDA_RESOURCE_DESC) -> CUresult;

    pub fn cuSurfObjectDestroy(surfObject: CUsurfObject) -> CUresult;

    pub fn cuSurfObjectGetResourceDesc(pResDesc: *mut CUDA_RESOURCE_DESC, surfObject: CUsurfObject) -> CUresult;

    pub fn cuDeviceCanAccessPeer(
        canAccessPeer: *mut ::std::os::raw::c_int,
        dev: CUdevice,
        peerDev: CUdevice,
    ) -> CUresult;

    pub fn cuDeviceGetP2PAttribute(
        value: *mut ::std::os::raw::c_int,
        attrib: CUdevice_P2PAttribute,
        srcDevice: CUdevice,
        dstDevice: CUdevice,
    ) -> CUresult;

    pub fn cuCtxEnablePeerAccess(peerContext: CUcontext, Flags: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuCtxDisablePeerAccess(peerContext: CUcontext) -> CUresult;

    pub fn cuGraphicsUnregisterResource(resource: CUgraphicsResource) -> CUresult;

    pub fn cuGraphicsSubResourceGetMappedArray(
        pArray: *mut CUarray,
        resource: CUgraphicsResource,
        arrayIndex: ::std::os::raw::c_uint,
        mipLevel: ::std::os::raw::c_uint,
    ) -> CUresult;

    pub fn cuGraphicsResourceGetMappedMipmappedArray(
        pMipmappedArray: *mut CUmipmappedArray,
        resource: CUgraphicsResource,
    ) -> CUresult;

    pub fn cuGraphicsResourceGetMappedPointer_v2(
        pDevPtr: *mut CUdeviceptr,
        pSize: *mut usize,
        resource: CUgraphicsResource,
    ) -> CUresult;

    pub fn cuGraphicsResourceSetMapFlags_v2(resource: CUgraphicsResource, flags: ::std::os::raw::c_uint) -> CUresult;

    pub fn cuGraphicsMapResources(
        count: ::std::os::raw::c_uint,
        resources: *mut CUgraphicsResource,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuGraphicsUnmapResources(
        count: ::std::os::raw::c_uint,
        resources: *mut CUgraphicsResource,
        hStream: CUstream,
    ) -> CUresult;

    pub fn cuGetExportTable(
        ppExportTable: *mut *const ::std::os::raw::c_void,
        pExportTableId: *const CUuuid,
    ) -> CUresult;
}
