#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

use vector_types::*;

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cudaError_t {
    Success = 0,
    MissingConfiguration = 1,
    MemoryAllocation = 2,
    InitializationError = 3,
    LaunchFailure = 4,
    PriorLaunchFailure = 5,
    LaunchTimeout = 6,
    LaunchOutOfResources = 7,
    InvalidDeviceFunction = 8,
    InvalidConfiguration = 9,
    InvalidDevice = 10,
    InvalidValue = 11,
    InvalidPitchValue = 12,
    InvalidSymbol = 13,
    MapBufferObjectFailed = 14,
    UnmapBufferObjectFailed = 15,
    InvalidHostPointer = 16,
    InvalidDevicePointer = 17,
    InvalidTexture = 18,
    InvalidTextureBinding = 19,
    InvalidChannelDescriptor = 20,
    InvalidMemcpyDirection = 21,
    AddressOfConstant = 22,
    TextureFetchFailed = 23,
    TextureNotBound = 24,
    SynchronizationError = 25,
    InvalidFilterSetting = 26,
    InvalidNormSetting = 27,
    MixedDeviceExecution = 28,
    CudartUnloading = 29,
    Unknown = 30,
    NotYetImplemented = 31,
    MemoryValueTooLarge = 32,
    InvalidResourceHandle = 33,
    NotReady = 34,
    InsufficientDriver = 35,
    SetOnActiveProcess = 36,
    InvalidSurface = 37,
    NoDevice = 38,
    ECCUncorrectable = 39,
    SharedObjectSymbolNotFound = 40,
    SharedObjectInitFailed = 41,
    UnsupportedLimit = 42,
    DuplicateVariableName = 43,
    DuplicateTextureName = 44,
    DuplicateSurfaceName = 45,
    DevicesUnavailable = 46,
    InvalidKernelImage = 47,
    NoKernelImageForDevice = 48,
    IncompatibleDriverContext = 49,
    PeerAccessAlreadyEnabled = 50,
    PeerAccessNotEnabled = 51,
    DeviceAlreadyInUse = 54,
    ProfilerDisabled = 55,
    ProfilerNotInitialized = 56,
    ProfilerAlreadyStarted = 57,
    ProfilerAlreadyStopped = 58,
    Assert = 59,
    TooManyPeers = 60,
    HostMemoryAlreadyRegistered = 61,
    HostMemoryNotRegistered = 62,
    OperatingSystem = 63,
    PeerAccessUnsupported = 64,
    LaunchMaxDepthExceeded = 65,
    LaunchFileScopedTex = 66,
    LaunchFileScopedSurf = 67,
    SyncDepthExceeded = 68,
    LaunchPendingCountExceeded = 69,
    NotPermitted = 70,
    NotSupported = 71,
    HardwareStackError = 72,
    IllegalInstruction = 73,
    MisalignedAddress = 74,
    InvalidAddressSpace = 75,
    InvalidPc = 76,
    IllegalAddress = 77,
    InvalidPtx = 78,
    InvalidGraphicsContext = 79,
    NvlinkUncorrectable = 80,
    JitCompilerNotFound = 81,
    CooperativeLaunchTooLarge = 82,
    StartupFailure = 127,
    ApiFailureBase = 10000,
}

pub const cudaHostAllocDefault: ::std::os::raw::c_uint = 0;
pub const cudaHostAllocPortable: ::std::os::raw::c_uint = 1;
pub const cudaHostAllocMapped: ::std::os::raw::c_uint = 2;
pub const cudaHostAllocWriteCombined: ::std::os::raw::c_uint = 4;
pub const cudaHostRegisterDefault: ::std::os::raw::c_uint = 0;
pub const cudaHostRegisterPortable: ::std::os::raw::c_uint = 1;
pub const cudaHostRegisterMapped: ::std::os::raw::c_uint = 2;
pub const cudaHostRegisterIoMemory: ::std::os::raw::c_uint = 4;
pub const cudaPeerAccessDefault: ::std::os::raw::c_uint = 0;
pub const cudaStreamDefault: ::std::os::raw::c_uint = 0;
pub const cudaStreamNonBlocking: ::std::os::raw::c_uint = 1;
pub const cudaEventDefault: ::std::os::raw::c_uint = 0;
pub const cudaEventBlockingSync: ::std::os::raw::c_uint = 1;
pub const cudaEventDisableTiming: ::std::os::raw::c_uint = 2;
pub const cudaEventInterprocess: ::std::os::raw::c_uint = 4;
pub const cudaDeviceScheduleAuto: ::std::os::raw::c_uint = 0;
pub const cudaDeviceScheduleSpin: ::std::os::raw::c_uint = 1;
pub const cudaDeviceScheduleYield: ::std::os::raw::c_uint = 2;
pub const cudaDeviceScheduleBlockingSync: ::std::os::raw::c_uint = 4;
pub const cudaDeviceBlockingSync: ::std::os::raw::c_uint = 4;
pub const cudaDeviceScheduleMask: ::std::os::raw::c_uint = 7;
pub const cudaDeviceMapHost: ::std::os::raw::c_uint = 8;
pub const cudaDeviceLmemResizeToMax: ::std::os::raw::c_uint = 16;
pub const cudaDeviceMask: ::std::os::raw::c_uint = 31;
pub const cudaArrayDefault: ::std::os::raw::c_uint = 0;
pub const cudaArrayLayered: ::std::os::raw::c_uint = 1;
pub const cudaArraySurfaceLoadStore: ::std::os::raw::c_uint = 2;
pub const cudaArrayCubemap: ::std::os::raw::c_uint = 4;
pub const cudaArrayTextureGather: ::std::os::raw::c_uint = 8;
pub const cudaIpcMemLazyEnablePeerAccess: ::std::os::raw::c_uint = 1;
pub const cudaMemAttachGlobal: ::std::os::raw::c_uint = 1;
pub const cudaMemAttachHost: ::std::os::raw::c_uint = 2;
pub const cudaMemAttachSingle: ::std::os::raw::c_uint = 4;
pub const cudaOccupancyDefault: ::std::os::raw::c_uint = 0;
pub const cudaOccupancyDisableCachingOverride: ::std::os::raw::c_uint = 1;
pub const cudaCooperativeLaunchMultiDeviceNoPreSync: ::std::os::raw::c_uint = 1;
pub const cudaCooperativeLaunchMultiDeviceNoPostSync: ::std::os::raw::c_uint = 2;
pub const CUDA_IPC_HANDLE_SIZE: ::std::os::raw::c_uint = 64;
pub const cudaSurfaceType1D: ::std::os::raw::c_uint = 1;
pub const cudaSurfaceType2D: ::std::os::raw::c_uint = 2;
pub const cudaSurfaceType3D: ::std::os::raw::c_uint = 3;
pub const cudaSurfaceTypeCubemap: ::std::os::raw::c_uint = 12;
pub const cudaSurfaceType1DLayered: ::std::os::raw::c_uint = 241;
pub const cudaSurfaceType2DLayered: ::std::os::raw::c_uint = 242;
pub const cudaSurfaceTypeCubemapLayered: ::std::os::raw::c_uint = 252;
pub const cudaTextureType1D: ::std::os::raw::c_uint = 1;
pub const cudaTextureType2D: ::std::os::raw::c_uint = 2;
pub const cudaTextureType3D: ::std::os::raw::c_uint = 3;
pub const cudaTextureTypeCubemap: ::std::os::raw::c_uint = 12;
pub const cudaTextureType1DLayered: ::std::os::raw::c_uint = 241;
pub const cudaTextureType2DLayered: ::std::os::raw::c_uint = 242;
pub const cudaTextureTypeCubemapLayered: ::std::os::raw::c_uint = 252;
pub const CUDART_VERSION: ::std::os::raw::c_uint = 9000;
pub const cudaRoundMode_cudaRoundNearest: cudaRoundMode = 0;
pub const cudaRoundMode_cudaRoundZero: cudaRoundMode = 1;
pub const cudaRoundMode_cudaRoundPosInf: cudaRoundMode = 2;
pub const cudaRoundMode_cudaRoundMinInf: cudaRoundMode = 3;
pub type cudaRoundMode = ::std::os::raw::c_uint;
pub const cudaChannelFormatKind_cudaChannelFormatKindSigned: cudaChannelFormatKind = 0;
pub const cudaChannelFormatKind_cudaChannelFormatKindUnsigned: cudaChannelFormatKind = 1;
pub const cudaChannelFormatKind_cudaChannelFormatKindFloat: cudaChannelFormatKind = 2;
pub const cudaChannelFormatKind_cudaChannelFormatKindNone: cudaChannelFormatKind = 3;
pub type cudaChannelFormatKind = ::std::os::raw::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaChannelFormatDesc {
    pub x: ::std::os::raw::c_int,
    pub y: ::std::os::raw::c_int,
    pub z: ::std::os::raw::c_int,
    pub w: ::std::os::raw::c_int,
    pub f: cudaChannelFormatKind,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaArray {
    _unused: [u8; 0],
}
pub type cudaArray_t = *mut cudaArray;
pub type cudaArray_const_t = *const cudaArray;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaMipmappedArray {
    _unused: [u8; 0],
}
pub type cudaMipmappedArray_t = *mut cudaMipmappedArray;
pub type cudaMipmappedArray_const_t = *const cudaMipmappedArray;
pub const cudaMemoryType_cudaMemoryTypeHost: cudaMemoryType = 1;
pub const cudaMemoryType_cudaMemoryTypeDevice: cudaMemoryType = 2;
pub type cudaMemoryType = ::std::os::raw::c_uint;
pub const cudaMemcpyKind_cudaMemcpyHostToHost: cudaMemcpyKind = 0;
pub const cudaMemcpyKind_cudaMemcpyHostToDevice: cudaMemcpyKind = 1;
pub const cudaMemcpyKind_cudaMemcpyDeviceToHost: cudaMemcpyKind = 2;
pub const cudaMemcpyKind_cudaMemcpyDeviceToDevice: cudaMemcpyKind = 3;
pub const cudaMemcpyKind_cudaMemcpyDefault: cudaMemcpyKind = 4;
pub type cudaMemcpyKind = ::std::os::raw::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaPitchedPtr {
    pub ptr: *mut ::std::os::raw::c_void,
    pub pitch: usize,
    pub xsize: usize,
    pub ysize: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaExtent {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaPos {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaMemcpy3DParms {
    pub srcArray: cudaArray_t,
    pub srcPos: cudaPos,
    pub srcPtr: cudaPitchedPtr,
    pub dstArray: cudaArray_t,
    pub dstPos: cudaPos,
    pub dstPtr: cudaPitchedPtr,
    pub extent: cudaExtent,
    pub kind: cudaMemcpyKind,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaMemcpy3DPeerParms {
    pub srcArray: cudaArray_t,
    pub srcPos: cudaPos,
    pub srcPtr: cudaPitchedPtr,
    pub srcDevice: ::std::os::raw::c_int,
    pub dstArray: cudaArray_t,
    pub dstPos: cudaPos,
    pub dstPtr: cudaPitchedPtr,
    pub dstDevice: ::std::os::raw::c_int,
    pub extent: cudaExtent,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaGraphicsResource {
    _unused: [u8; 0],
}
pub const cudaGraphicsRegisterFlags_cudaGraphicsRegisterFlagsNone: cudaGraphicsRegisterFlags = 0;
pub const cudaGraphicsRegisterFlags_cudaGraphicsRegisterFlagsReadOnly: cudaGraphicsRegisterFlags = 1;
pub const cudaGraphicsRegisterFlags_cudaGraphicsRegisterFlagsWriteDiscard: cudaGraphicsRegisterFlags = 2;
pub const cudaGraphicsRegisterFlags_cudaGraphicsRegisterFlagsSurfaceLoadStore: cudaGraphicsRegisterFlags = 4;
pub const cudaGraphicsRegisterFlags_cudaGraphicsRegisterFlagsTextureGather: cudaGraphicsRegisterFlags = 8;
pub type cudaGraphicsRegisterFlags = ::std::os::raw::c_uint;
pub const cudaGraphicsMapFlags_cudaGraphicsMapFlagsNone: cudaGraphicsMapFlags = 0;
pub const cudaGraphicsMapFlags_cudaGraphicsMapFlagsReadOnly: cudaGraphicsMapFlags = 1;
pub const cudaGraphicsMapFlags_cudaGraphicsMapFlagsWriteDiscard: cudaGraphicsMapFlags = 2;
pub type cudaGraphicsMapFlags = ::std::os::raw::c_uint;
pub const cudaGraphicsCubeFace_cudaGraphicsCubeFacePositiveX: cudaGraphicsCubeFace = 0;
pub const cudaGraphicsCubeFace_cudaGraphicsCubeFaceNegativeX: cudaGraphicsCubeFace = 1;
pub const cudaGraphicsCubeFace_cudaGraphicsCubeFacePositiveY: cudaGraphicsCubeFace = 2;
pub const cudaGraphicsCubeFace_cudaGraphicsCubeFaceNegativeY: cudaGraphicsCubeFace = 3;
pub const cudaGraphicsCubeFace_cudaGraphicsCubeFacePositiveZ: cudaGraphicsCubeFace = 4;
pub const cudaGraphicsCubeFace_cudaGraphicsCubeFaceNegativeZ: cudaGraphicsCubeFace = 5;
pub type cudaGraphicsCubeFace = ::std::os::raw::c_uint;
pub const cudaResourceType_cudaResourceTypeArray: cudaResourceType = 0;
pub const cudaResourceType_cudaResourceTypeMipmappedArray: cudaResourceType = 1;
pub const cudaResourceType_cudaResourceTypeLinear: cudaResourceType = 2;
pub const cudaResourceType_cudaResourceTypePitch2D: cudaResourceType = 3;
pub type cudaResourceType = ::std::os::raw::c_uint;
pub const cudaResourceViewFormat_cudaResViewFormatNone: cudaResourceViewFormat = 0;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedChar1: cudaResourceViewFormat = 1;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedChar2: cudaResourceViewFormat = 2;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedChar4: cudaResourceViewFormat = 3;
pub const cudaResourceViewFormat_cudaResViewFormatSignedChar1: cudaResourceViewFormat = 4;
pub const cudaResourceViewFormat_cudaResViewFormatSignedChar2: cudaResourceViewFormat = 5;
pub const cudaResourceViewFormat_cudaResViewFormatSignedChar4: cudaResourceViewFormat = 6;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedShort1: cudaResourceViewFormat = 7;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedShort2: cudaResourceViewFormat = 8;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedShort4: cudaResourceViewFormat = 9;
pub const cudaResourceViewFormat_cudaResViewFormatSignedShort1: cudaResourceViewFormat = 10;
pub const cudaResourceViewFormat_cudaResViewFormatSignedShort2: cudaResourceViewFormat = 11;
pub const cudaResourceViewFormat_cudaResViewFormatSignedShort4: cudaResourceViewFormat = 12;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedInt1: cudaResourceViewFormat = 13;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedInt2: cudaResourceViewFormat = 14;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedInt4: cudaResourceViewFormat = 15;
pub const cudaResourceViewFormat_cudaResViewFormatSignedInt1: cudaResourceViewFormat = 16;
pub const cudaResourceViewFormat_cudaResViewFormatSignedInt2: cudaResourceViewFormat = 17;
pub const cudaResourceViewFormat_cudaResViewFormatSignedInt4: cudaResourceViewFormat = 18;
pub const cudaResourceViewFormat_cudaResViewFormatHalf1: cudaResourceViewFormat = 19;
pub const cudaResourceViewFormat_cudaResViewFormatHalf2: cudaResourceViewFormat = 20;
pub const cudaResourceViewFormat_cudaResViewFormatHalf4: cudaResourceViewFormat = 21;
pub const cudaResourceViewFormat_cudaResViewFormatFloat1: cudaResourceViewFormat = 22;
pub const cudaResourceViewFormat_cudaResViewFormatFloat2: cudaResourceViewFormat = 23;
pub const cudaResourceViewFormat_cudaResViewFormatFloat4: cudaResourceViewFormat = 24;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedBlockCompressed1: cudaResourceViewFormat = 25;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedBlockCompressed2: cudaResourceViewFormat = 26;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedBlockCompressed3: cudaResourceViewFormat = 27;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedBlockCompressed4: cudaResourceViewFormat = 28;
pub const cudaResourceViewFormat_cudaResViewFormatSignedBlockCompressed4: cudaResourceViewFormat = 29;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedBlockCompressed5: cudaResourceViewFormat = 30;
pub const cudaResourceViewFormat_cudaResViewFormatSignedBlockCompressed5: cudaResourceViewFormat = 31;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedBlockCompressed6H: cudaResourceViewFormat = 32;
pub const cudaResourceViewFormat_cudaResViewFormatSignedBlockCompressed6H: cudaResourceViewFormat = 33;
pub const cudaResourceViewFormat_cudaResViewFormatUnsignedBlockCompressed7: cudaResourceViewFormat = 34;
pub type cudaResourceViewFormat = ::std::os::raw::c_uint;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaResourceDesc {
    pub resType: cudaResourceType,
    pub res: cudaResourceDesc__bindgen_ty_1,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaResourceDesc__bindgen_ty_1 {
    pub array: cudaResourceDesc__bindgen_ty_1__bindgen_ty_1,
    pub mipmap: cudaResourceDesc__bindgen_ty_1__bindgen_ty_2,
    pub linear: cudaResourceDesc__bindgen_ty_1__bindgen_ty_3,
    pub pitch2D: cudaResourceDesc__bindgen_ty_1__bindgen_ty_4,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_1 {
    pub array: cudaArray_t,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_2 {
    pub mipmap: cudaMipmappedArray_t,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_3 {
    pub devPtr: *mut ::std::os::raw::c_void,
    pub desc: cudaChannelFormatDesc,
    pub sizeInBytes: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_4 {
    pub devPtr: *mut ::std::os::raw::c_void,
    pub desc: cudaChannelFormatDesc,
    pub width: usize,
    pub height: usize,
    pub pitchInBytes: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaResourceViewDesc {
    pub format: cudaResourceViewFormat,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub firstMipmapLevel: ::std::os::raw::c_uint,
    pub lastMipmapLevel: ::std::os::raw::c_uint,
    pub firstLayer: ::std::os::raw::c_uint,
    pub lastLayer: ::std::os::raw::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaPointerAttributes {
    pub memoryType: cudaMemoryType,
    pub device: ::std::os::raw::c_int,
    pub devicePointer: *mut ::std::os::raw::c_void,
    pub hostPointer: *mut ::std::os::raw::c_void,
    pub isManaged: ::std::os::raw::c_int,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaFuncAttributes {
    pub sharedSizeBytes: usize,
    pub constSizeBytes: usize,
    pub localSizeBytes: usize,
    pub maxThreadsPerBlock: ::std::os::raw::c_int,
    pub numRegs: ::std::os::raw::c_int,
    pub ptxVersion: ::std::os::raw::c_int,
    pub binaryVersion: ::std::os::raw::c_int,
    pub cacheModeCA: ::std::os::raw::c_int,
    pub maxDynamicSharedSizeBytes: ::std::os::raw::c_int,
    pub preferredShmemCarveout: ::std::os::raw::c_int,
}
pub const cudaFuncAttribute_cudaFuncAttributeMaxDynamicSharedMemorySize: cudaFuncAttribute = 8;
pub const cudaFuncAttribute_cudaFuncAttributePreferredSharedMemoryCarveout: cudaFuncAttribute = 9;
pub const cudaFuncAttribute_cudaFuncAttributeMax: cudaFuncAttribute = 10;
pub type cudaFuncAttribute = ::std::os::raw::c_uint;
pub const cudaFuncCache_cudaFuncCachePreferNone: cudaFuncCache = 0;
pub const cudaFuncCache_cudaFuncCachePreferShared: cudaFuncCache = 1;
pub const cudaFuncCache_cudaFuncCachePreferL1: cudaFuncCache = 2;
pub const cudaFuncCache_cudaFuncCachePreferEqual: cudaFuncCache = 3;
pub type cudaFuncCache = ::std::os::raw::c_uint;
pub const cudaSharedMemConfig_cudaSharedMemBankSizeDefault: cudaSharedMemConfig = 0;
pub const cudaSharedMemConfig_cudaSharedMemBankSizeFourByte: cudaSharedMemConfig = 1;
pub const cudaSharedMemConfig_cudaSharedMemBankSizeEightByte: cudaSharedMemConfig = 2;
pub type cudaSharedMemConfig = ::std::os::raw::c_uint;
pub const cudaSharedCarveout_cudaSharedmemCarveoutDefault: cudaSharedCarveout = -1;
pub const cudaSharedCarveout_cudaSharedmemCarveoutMaxShared: cudaSharedCarveout = 100;
pub const cudaSharedCarveout_cudaSharedmemCarveoutMaxL1: cudaSharedCarveout = 0;
pub type cudaSharedCarveout = ::std::os::raw::c_int;
pub const cudaComputeMode_cudaComputeModeDefault: cudaComputeMode = 0;
pub const cudaComputeMode_cudaComputeModeExclusive: cudaComputeMode = 1;
pub const cudaComputeMode_cudaComputeModeProhibited: cudaComputeMode = 2;
pub const cudaComputeMode_cudaComputeModeExclusiveProcess: cudaComputeMode = 3;
pub type cudaComputeMode = ::std::os::raw::c_uint;
pub const cudaLimit_cudaLimitStackSize: cudaLimit = 0;
pub const cudaLimit_cudaLimitPrintfFifoSize: cudaLimit = 1;
pub const cudaLimit_cudaLimitMallocHeapSize: cudaLimit = 2;
pub const cudaLimit_cudaLimitDevRuntimeSyncDepth: cudaLimit = 3;
pub const cudaLimit_cudaLimitDevRuntimePendingLaunchCount: cudaLimit = 4;
pub type cudaLimit = ::std::os::raw::c_uint;
pub const cudaMemoryAdvise_cudaMemAdviseSetReadMostly: cudaMemoryAdvise = 1;
pub const cudaMemoryAdvise_cudaMemAdviseUnsetReadMostly: cudaMemoryAdvise = 2;
pub const cudaMemoryAdvise_cudaMemAdviseSetPreferredLocation: cudaMemoryAdvise = 3;
pub const cudaMemoryAdvise_cudaMemAdviseUnsetPreferredLocation: cudaMemoryAdvise = 4;
pub const cudaMemoryAdvise_cudaMemAdviseSetAccessedBy: cudaMemoryAdvise = 5;
pub const cudaMemoryAdvise_cudaMemAdviseUnsetAccessedBy: cudaMemoryAdvise = 6;
pub type cudaMemoryAdvise = ::std::os::raw::c_uint;
pub const cudaMemRangeAttribute_cudaMemRangeAttributeReadMostly: cudaMemRangeAttribute = 1;
pub const cudaMemRangeAttribute_cudaMemRangeAttributePreferredLocation: cudaMemRangeAttribute = 2;
pub const cudaMemRangeAttribute_cudaMemRangeAttributeAccessedBy: cudaMemRangeAttribute = 3;
pub const cudaMemRangeAttribute_cudaMemRangeAttributeLastPrefetchLocation: cudaMemRangeAttribute = 4;
pub type cudaMemRangeAttribute = ::std::os::raw::c_uint;
pub const cudaOutputMode_cudaKeyValuePair: cudaOutputMode = 0;
pub const cudaOutputMode_cudaCSV: cudaOutputMode = 1;
pub type cudaOutputMode = ::std::os::raw::c_uint;
pub const cudaDeviceP2PAttr_cudaDevP2PAttrPerformanceRank: cudaDeviceP2PAttr = 1;
pub const cudaDeviceP2PAttr_cudaDevP2PAttrAccessSupported: cudaDeviceP2PAttr = 2;
pub const cudaDeviceP2PAttr_cudaDevP2PAttrNativeAtomicSupported: cudaDeviceP2PAttr = 3;
pub type cudaDeviceP2PAttr = ::std::os::raw::c_uint;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaDeviceProp {
    pub name: [::std::os::raw::c_char; 256usize],
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: ::std::os::raw::c_int,
    pub warpSize: ::std::os::raw::c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: ::std::os::raw::c_int,
    pub maxThreadsDim: [::std::os::raw::c_int; 3usize],
    pub maxGridSize: [::std::os::raw::c_int; 3usize],
    pub clockRate: ::std::os::raw::c_int,
    pub totalConstMem: usize,
    pub major: ::std::os::raw::c_int,
    pub minor: ::std::os::raw::c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: ::std::os::raw::c_int,
    pub multiProcessorCount: ::std::os::raw::c_int,
    pub kernelExecTimeoutEnabled: ::std::os::raw::c_int,
    pub integrated: ::std::os::raw::c_int,
    pub canMapHostMemory: ::std::os::raw::c_int,
    pub computeMode: ::std::os::raw::c_int,
    pub maxTexture1D: ::std::os::raw::c_int,
    pub maxTexture1DMipmap: ::std::os::raw::c_int,
    pub maxTexture1DLinear: ::std::os::raw::c_int,
    pub maxTexture2D: [::std::os::raw::c_int; 2usize],
    pub maxTexture2DMipmap: [::std::os::raw::c_int; 2usize],
    pub maxTexture2DLinear: [::std::os::raw::c_int; 3usize],
    pub maxTexture2DGather: [::std::os::raw::c_int; 2usize],
    pub maxTexture3D: [::std::os::raw::c_int; 3usize],
    pub maxTexture3DAlt: [::std::os::raw::c_int; 3usize],
    pub maxTextureCubemap: ::std::os::raw::c_int,
    pub maxTexture1DLayered: [::std::os::raw::c_int; 2usize],
    pub maxTexture2DLayered: [::std::os::raw::c_int; 3usize],
    pub maxTextureCubemapLayered: [::std::os::raw::c_int; 2usize],
    pub maxSurface1D: ::std::os::raw::c_int,
    pub maxSurface2D: [::std::os::raw::c_int; 2usize],
    pub maxSurface3D: [::std::os::raw::c_int; 3usize],
    pub maxSurface1DLayered: [::std::os::raw::c_int; 2usize],
    pub maxSurface2DLayered: [::std::os::raw::c_int; 3usize],
    pub maxSurfaceCubemap: ::std::os::raw::c_int,
    pub maxSurfaceCubemapLayered: [::std::os::raw::c_int; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: ::std::os::raw::c_int,
    pub ECCEnabled: ::std::os::raw::c_int,
    pub pciBusID: ::std::os::raw::c_int,
    pub pciDeviceID: ::std::os::raw::c_int,
    pub pciDomainID: ::std::os::raw::c_int,
    pub tccDriver: ::std::os::raw::c_int,
    pub asyncEngineCount: ::std::os::raw::c_int,
    pub unifiedAddressing: ::std::os::raw::c_int,
    pub memoryClockRate: ::std::os::raw::c_int,
    pub memoryBusWidth: ::std::os::raw::c_int,
    pub l2CacheSize: ::std::os::raw::c_int,
    pub maxThreadsPerMultiProcessor: ::std::os::raw::c_int,
    pub streamPrioritiesSupported: ::std::os::raw::c_int,
    pub globalL1CacheSupported: ::std::os::raw::c_int,
    pub localL1CacheSupported: ::std::os::raw::c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: ::std::os::raw::c_int,
    pub managedMemory: ::std::os::raw::c_int,
    pub isMultiGpuBoard: ::std::os::raw::c_int,
    pub multiGpuBoardGroupID: ::std::os::raw::c_int,
    pub hostNativeAtomicSupported: ::std::os::raw::c_int,
    pub singleToDoublePrecisionPerfRatio: ::std::os::raw::c_int,
    pub pageableMemoryAccess: ::std::os::raw::c_int,
    pub concurrentManagedAccess: ::std::os::raw::c_int,
    pub computePreemptionSupported: ::std::os::raw::c_int,
    pub canUseHostPointerForRegisteredMem: ::std::os::raw::c_int,
    pub cooperativeLaunch: ::std::os::raw::c_int,
    pub cooperativeMultiDeviceLaunch: ::std::os::raw::c_int,
    pub sharedMemPerBlockOptin: usize,
}

#[repr(u32)]
#[derive(Copy, Clone)]
pub enum cudaDeviceAttr {
    MaxThreadsPerBlock = 1,
    MaxBlockDimX = 2,
    MaxBlockDimY = 3,
    MaxBlockDimZ = 4,
    MaxGridDimX = 5,
    MaxGridDimY = 6,
    MaxGridDimZ = 7,
    MaxSharedMemoryPerBlock = 8,
    TotalConstantMemory = 9,
    WarpSize = 10,
    MaxPitch = 11,
    MaxRegistersPerBlock = 12,
    ClockRate = 13,
    TextureAlignment = 14,
    GpuOverlap = 15,
    MultiProcessorCount = 16,
    KernelExecTimeout = 17,
    Integrated = 18,
    CanMapHostMemory = 19,
    ComputeMode = 20,
    MaxTexture1DWidth = 21,
    MaxTexture2DWidth = 22,
    MaxTexture2DHeight = 23,
    MaxTexture3DWidth = 24,
    MaxTexture3DHeight = 25,
    MaxTexture3DDepth = 26,
    MaxTexture2DLayeredWidth = 27,
    MaxTexture2DLayeredHeight = 28,
    MaxTexture2DLayeredLayers = 29,
    SurfaceAlignment = 30,
    ConcurrentKernels = 31,
    EccEnabled = 32,
    PciBusId = 33,
    PciDeviceId = 34,
    TccDriver = 35,
    MemoryClockRate = 36,
    GlobalMemoryBusWidth = 37,
    L2CacheSize = 38,
    MaxThreadsPerMultiProcessor = 39,
    AsyncEngineCount = 40,
    UnifiedAddressing = 41,
    MaxTexture1DLayeredWidth = 42,
    MaxTexture1DLayeredLayers = 43,
    MaxTexture2DGatherWidth = 45,
    MaxTexture2DGatherHeight = 46,
    MaxTexture3DWidthAlt = 47,
    MaxTexture3DHeightAlt = 48,
    MaxTexture3DDepthAlt = 49,
    PciDomainId = 50,
    TexturePitchAlignment = 51,
    MaxTextureCubemapWidth = 52,
    MaxTextureCubemapLayeredWidth = 53,
    MaxTextureCubemapLayeredLayers = 54,
    MaxSurface1DWidth = 55,
    MaxSurface2DWidth = 56,
    MaxSurface2DHeight = 57,
    MaxSurface3DWidth = 58,
    MaxSurface3DHeight = 59,
    MaxSurface3DDepth = 60,
    MaxSurface1DLayeredWidth = 61,
    MaxSurface1DLayeredLayers = 62,
    MaxSurface2DLayeredWidth = 63,
    MaxSurface2DLayeredHeight = 64,
    MaxSurface2DLayeredLayers = 65,
    MaxSurfaceCubemapWidth = 66,
    MaxSurfaceCubemapLayeredWidth = 67,
    MaxSurfaceCubemapLayeredLayers = 68,
    MaxTexture1DLinearWidth = 69,
    MaxTexture2DLinearWidth = 70,
    MaxTexture2DLinearHeight = 71,
    MaxTexture2DLinearPitch = 72,
    MaxTexture2DMipmappedWidth = 73,
    MaxTexture2DMipmappedHeight = 74,
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76,
    MaxTexture1DMipmappedWidth = 77,
    StreamPrioritiesSupported = 78,
    GlobalL1CacheSupported = 79,
    LocalL1CacheSupported = 80,
    MaxSharedMemoryPerMultiprocessor = 81,
    MaxRegistersPerMultiprocessor = 82,
    ManagedMemory = 83,
    IsMultiGpuBoard = 84,
    MultiGpuBoardGroupID = 85,
    HostNativeAtomicSupported = 86,
    SingleToDoublePrecisionPerfRatio = 87,
    PageableMemoryAccess = 88,
    ConcurrentManagedAccess = 89,
    ComputePreemptionSupported = 90,
    CanUseHostPointerForRegisteredMem = 91,
    Reserved92 = 92,
    Reserved93 = 93,
    Reserved94 = 94,
    CooperativeLaunch = 95,
    CooperativeMultiDeviceLaunch = 96,
    MaxSharedMemoryPerBlockOptin = 97,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaIpcEventHandle_st {
    pub reserved: [::std::os::raw::c_char; 64usize],
}
pub type cudaIpcEventHandle_t = cudaIpcEventHandle_st;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaIpcMemHandle_st {
    pub reserved: [::std::os::raw::c_char; 64usize],
}
pub type cudaIpcMemHandle_t = cudaIpcMemHandle_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
pub type cudaStream_t = *mut CUstream_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUevent_st {
    _unused: [u8; 0],
}
pub type cudaEvent_t = *mut CUevent_st;
pub type cudaGraphicsResource_t = *mut cudaGraphicsResource;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUuuid_st {
    _unused: [u8; 0],
}
pub type cudaUUID_t = CUuuid_st;
pub use self::cudaOutputMode as cudaOutputMode_t;
pub const cudaCGScope_cudaCGScopeInvalid: cudaCGScope = 0;
pub const cudaCGScope_cudaCGScopeGrid: cudaCGScope = 1;
pub const cudaCGScope_cudaCGScopeMultiGrid: cudaCGScope = 2;
pub type cudaCGScope = ::std::os::raw::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaLaunchParams {
    pub func: *mut ::std::os::raw::c_void,
    pub gridDim: dim3,
    pub blockDim: dim3,
    pub args: *mut *mut ::std::os::raw::c_void,
    pub sharedMem: usize,
    pub stream: cudaStream_t,
}
pub const cudaSurfaceBoundaryMode_cudaBoundaryModeZero: cudaSurfaceBoundaryMode = 0;
pub const cudaSurfaceBoundaryMode_cudaBoundaryModeClamp: cudaSurfaceBoundaryMode = 1;
pub const cudaSurfaceBoundaryMode_cudaBoundaryModeTrap: cudaSurfaceBoundaryMode = 2;
pub type cudaSurfaceBoundaryMode = ::std::os::raw::c_uint;
pub const cudaSurfaceFormatMode_cudaFormatModeForced: cudaSurfaceFormatMode = 0;
pub const cudaSurfaceFormatMode_cudaFormatModeAuto: cudaSurfaceFormatMode = 1;
pub type cudaSurfaceFormatMode = ::std::os::raw::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct surfaceReference {
    pub channelDesc: cudaChannelFormatDesc,
}
pub type cudaSurfaceObject_t = ::std::os::raw::c_ulonglong;
pub const cudaTextureAddressMode_cudaAddressModeWrap: cudaTextureAddressMode = 0;
pub const cudaTextureAddressMode_cudaAddressModeClamp: cudaTextureAddressMode = 1;
pub const cudaTextureAddressMode_cudaAddressModeMirror: cudaTextureAddressMode = 2;
pub const cudaTextureAddressMode_cudaAddressModeBorder: cudaTextureAddressMode = 3;
pub type cudaTextureAddressMode = ::std::os::raw::c_uint;
pub const cudaTextureFilterMode_cudaFilterModePoint: cudaTextureFilterMode = 0;
pub const cudaTextureFilterMode_cudaFilterModeLinear: cudaTextureFilterMode = 1;
pub type cudaTextureFilterMode = ::std::os::raw::c_uint;
pub const cudaTextureReadMode_cudaReadModeElementType: cudaTextureReadMode = 0;
pub const cudaTextureReadMode_cudaReadModeNormalizedFloat: cudaTextureReadMode = 1;
pub type cudaTextureReadMode = ::std::os::raw::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct textureReference {
    pub normalized: ::std::os::raw::c_int,
    pub filterMode: cudaTextureFilterMode,
    pub addressMode: [cudaTextureAddressMode; 3usize],
    pub channelDesc: cudaChannelFormatDesc,
    pub sRGB: ::std::os::raw::c_int,
    pub maxAnisotropy: ::std::os::raw::c_uint,
    pub mipmapFilterMode: cudaTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub __cudaReserved: [::std::os::raw::c_int; 15usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaTextureDesc {
    pub addressMode: [cudaTextureAddressMode; 3usize],
    pub filterMode: cudaTextureFilterMode,
    pub readMode: cudaTextureReadMode,
    pub sRGB: ::std::os::raw::c_int,
    pub borderColor: [f32; 4usize],
    pub normalizedCoords: ::std::os::raw::c_int,
    pub maxAnisotropy: ::std::os::raw::c_uint,
    pub mipmapFilterMode: cudaTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
}
pub type cudaTextureObject_t = ::std::os::raw::c_ulonglong;
pub const cudaDataType_t_CUDA_R_16F: cudaDataType_t = 2;
pub const cudaDataType_t_CUDA_C_16F: cudaDataType_t = 6;
pub const cudaDataType_t_CUDA_R_32F: cudaDataType_t = 0;
pub const cudaDataType_t_CUDA_C_32F: cudaDataType_t = 4;
pub const cudaDataType_t_CUDA_R_64F: cudaDataType_t = 1;
pub const cudaDataType_t_CUDA_C_64F: cudaDataType_t = 5;
pub const cudaDataType_t_CUDA_R_8I: cudaDataType_t = 3;
pub const cudaDataType_t_CUDA_C_8I: cudaDataType_t = 7;
pub const cudaDataType_t_CUDA_R_8U: cudaDataType_t = 8;
pub const cudaDataType_t_CUDA_C_8U: cudaDataType_t = 9;
pub const cudaDataType_t_CUDA_R_32I: cudaDataType_t = 10;
pub const cudaDataType_t_CUDA_C_32I: cudaDataType_t = 11;
pub const cudaDataType_t_CUDA_R_32U: cudaDataType_t = 12;
pub const cudaDataType_t_CUDA_C_32U: cudaDataType_t = 13;
pub type cudaDataType_t = ::std::os::raw::c_uint;
pub use self::cudaDataType_t as cudaDataType;
pub const libraryPropertyType_t_MAJOR_VERSION: libraryPropertyType_t = 0;
pub const libraryPropertyType_t_MINOR_VERSION: libraryPropertyType_t = 1;
pub const libraryPropertyType_t_PATCH_LEVEL: libraryPropertyType_t = 2;
pub type libraryPropertyType_t = ::std::os::raw::c_uint;
pub use self::libraryPropertyType_t as libraryPropertyType;
extern "C" {
    pub fn cudaDeviceReset() -> cudaError_t;

    pub fn cudaDeviceSynchronize() -> cudaError_t;

    pub fn cudaDeviceSetLimit(limit: cudaLimit, value: usize) -> cudaError_t;

    pub fn cudaDeviceGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t;

    pub fn cudaDeviceGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t;

    pub fn cudaDeviceGetStreamPriorityRange(
        leastPriority: *mut ::std::os::raw::c_int,
        greatestPriority: *mut ::std::os::raw::c_int,
    ) -> cudaError_t;

    pub fn cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t;

    pub fn cudaDeviceGetSharedMemConfig(pConfig: *mut cudaSharedMemConfig) -> cudaError_t;

    pub fn cudaDeviceSetSharedMemConfig(config: cudaSharedMemConfig) -> cudaError_t;

    pub fn cudaDeviceGetByPCIBusId(
        device: *mut ::std::os::raw::c_int,
        pciBusId: *const ::std::os::raw::c_char,
    ) -> cudaError_t;

    pub fn cudaDeviceGetPCIBusId(
        pciBusId: *mut ::std::os::raw::c_char,
        len: ::std::os::raw::c_int,
        device: ::std::os::raw::c_int,
    ) -> cudaError_t;

    pub fn cudaIpcGetEventHandle(handle: *mut cudaIpcEventHandle_t, event: cudaEvent_t) -> cudaError_t;

    pub fn cudaIpcOpenEventHandle(event: *mut cudaEvent_t, handle: cudaIpcEventHandle_t) -> cudaError_t;

    pub fn cudaIpcGetMemHandle(handle: *mut cudaIpcMemHandle_t, devPtr: *mut ::std::os::raw::c_void) -> cudaError_t;

    pub fn cudaIpcOpenMemHandle(
        devPtr: *mut *mut ::std::os::raw::c_void,
        handle: cudaIpcMemHandle_t,
        flags: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaIpcCloseMemHandle(devPtr: *mut ::std::os::raw::c_void) -> cudaError_t;

    pub fn cudaThreadExit() -> cudaError_t;

    pub fn cudaThreadSynchronize() -> cudaError_t;

    pub fn cudaThreadSetLimit(limit: cudaLimit, value: usize) -> cudaError_t;

    pub fn cudaThreadGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t;

    pub fn cudaThreadGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t;

    pub fn cudaThreadSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t;

    pub fn cudaGetLastError() -> cudaError_t;

    pub fn cudaPeekAtLastError() -> cudaError_t;

    pub fn cudaGetErrorName(error: cudaError_t) -> *const ::std::os::raw::c_char;

    pub fn cudaGetErrorString(error: cudaError_t) -> *const ::std::os::raw::c_char;

    pub fn cudaGetDeviceCount(count: *mut ::std::os::raw::c_int) -> cudaError_t;

    pub fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: ::std::os::raw::c_int) -> cudaError_t;

    pub fn cudaDeviceGetAttribute(
        value: *mut ::std::os::raw::c_int,
        attr: cudaDeviceAttr,
        device: ::std::os::raw::c_int,
    ) -> cudaError_t;

    pub fn cudaDeviceGetP2PAttribute(
        value: *mut ::std::os::raw::c_int,
        attr: cudaDeviceP2PAttr,
        srcDevice: ::std::os::raw::c_int,
        dstDevice: ::std::os::raw::c_int,
    ) -> cudaError_t;

    pub fn cudaChooseDevice(device: *mut ::std::os::raw::c_int, prop: *const cudaDeviceProp) -> cudaError_t;

    pub fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t;

    pub fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t;

    pub fn cudaSetValidDevices(device_arr: *mut ::std::os::raw::c_int, len: ::std::os::raw::c_int) -> cudaError_t;

    pub fn cudaSetDeviceFlags(flags: ::std::os::raw::c_uint) -> cudaError_t;

    pub fn cudaGetDeviceFlags(flags: *mut ::std::os::raw::c_uint) -> cudaError_t;

    pub fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;

    pub fn cudaStreamCreateWithFlags(pStream: *mut cudaStream_t, flags: ::std::os::raw::c_uint) -> cudaError_t;

    pub fn cudaStreamCreateWithPriority(
        pStream: *mut cudaStream_t,
        flags: ::std::os::raw::c_uint,
        priority: ::std::os::raw::c_int,
    ) -> cudaError_t;

    pub fn cudaStreamGetPriority(hStream: cudaStream_t, priority: *mut ::std::os::raw::c_int) -> cudaError_t;

    pub fn cudaStreamGetFlags(hStream: cudaStream_t, flags: *mut ::std::os::raw::c_uint) -> cudaError_t;

    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;

    pub fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: ::std::os::raw::c_uint) -> cudaError_t;
}
pub type cudaStreamCallback_t = ::std::option::Option<
    unsafe extern "C" fn(stream: cudaStream_t,
                         status: cudaError_t,
                         userData: *mut ::std::os::raw::c_void),
>;
extern "C" {
    pub fn cudaStreamAddCallback(
        stream: cudaStream_t,
        callback: cudaStreamCallback_t,
        userData: *mut ::std::os::raw::c_void,
        flags: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;

    pub fn cudaStreamQuery(stream: cudaStream_t) -> cudaError_t;

    pub fn cudaStreamAttachMemAsync(
        stream: cudaStream_t,
        devPtr: *mut ::std::os::raw::c_void,
        length: usize,
        flags: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;

    pub fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: ::std::os::raw::c_uint) -> cudaError_t;

    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;

    pub fn cudaEventQuery(event: cudaEvent_t) -> cudaError_t;

    pub fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;

    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;

    pub fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;

    pub fn cudaLaunchKernel(
        func: *const ::std::os::raw::c_void,
        gridDim: dim3,
        blockDim: dim3,
        args: *mut *mut ::std::os::raw::c_void,
        sharedMem: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaLaunchCooperativeKernel(
        func: *const ::std::os::raw::c_void,
        gridDim: dim3,
        blockDim: dim3,
        args: *mut *mut ::std::os::raw::c_void,
        sharedMem: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaLaunchCooperativeKernelMultiDevice(
        launchParamsList: *mut cudaLaunchParams,
        numDevices: ::std::os::raw::c_uint,
        flags: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaFuncSetCacheConfig(func: *const ::std::os::raw::c_void, cacheConfig: cudaFuncCache) -> cudaError_t;

    pub fn cudaFuncSetSharedMemConfig(func: *const ::std::os::raw::c_void, config: cudaSharedMemConfig) -> cudaError_t;

    pub fn cudaFuncGetAttributes(attr: *mut cudaFuncAttributes, func: *const ::std::os::raw::c_void) -> cudaError_t;

    pub fn cudaFuncSetAttribute(
        func: *const ::std::os::raw::c_void,
        attr: cudaFuncAttribute,
        value: ::std::os::raw::c_int,
    ) -> cudaError_t;

    pub fn cudaSetDoubleForDevice(d: *mut f64) -> cudaError_t;

    pub fn cudaSetDoubleForHost(d: *mut f64) -> cudaError_t;

    pub fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks: *mut ::std::os::raw::c_int,
        func: *const ::std::os::raw::c_void,
        blockSize: ::std::os::raw::c_int,
        dynamicSMemSize: usize,
    ) -> cudaError_t;

    pub fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks: *mut ::std::os::raw::c_int,
        func: *const ::std::os::raw::c_void,
        blockSize: ::std::os::raw::c_int,
        dynamicSMemSize: usize,
        flags: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaConfigureCall(gridDim: dim3, blockDim: dim3, sharedMem: usize, stream: cudaStream_t) -> cudaError_t;

    pub fn cudaSetupArgument(arg: *const ::std::os::raw::c_void, size: usize, offset: usize) -> cudaError_t;

    pub fn cudaLaunch(func: *const ::std::os::raw::c_void) -> cudaError_t;

    pub fn cudaMallocManaged(
        devPtr: *mut *mut ::std::os::raw::c_void,
        size: usize,
        flags: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaMalloc(devPtr: *mut *mut ::std::os::raw::c_void, size: usize) -> cudaError_t;

    pub fn cudaMallocHost(ptr: *mut *mut ::std::os::raw::c_void, size: usize) -> cudaError_t;

    pub fn cudaMallocPitch(
        devPtr: *mut *mut ::std::os::raw::c_void,
        pitch: *mut usize,
        width: usize,
        height: usize,
    ) -> cudaError_t;

    pub fn cudaMallocArray(
        array: *mut cudaArray_t,
        desc: *const cudaChannelFormatDesc,
        width: usize,
        height: usize,
        flags: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaFree(devPtr: *mut ::std::os::raw::c_void) -> cudaError_t;

    pub fn cudaFreeHost(ptr: *mut ::std::os::raw::c_void) -> cudaError_t;

    pub fn cudaFreeArray(array: cudaArray_t) -> cudaError_t;

    pub fn cudaFreeMipmappedArray(mipmappedArray: cudaMipmappedArray_t) -> cudaError_t;

    pub fn cudaHostAlloc(
        pHost: *mut *mut ::std::os::raw::c_void,
        size: usize,
        flags: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaHostRegister(
        ptr: *mut ::std::os::raw::c_void,
        size: usize,
        flags: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaHostUnregister(ptr: *mut ::std::os::raw::c_void) -> cudaError_t;

    pub fn cudaHostGetDevicePointer(
        pDevice: *mut *mut ::std::os::raw::c_void,
        pHost: *mut ::std::os::raw::c_void,
        flags: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaHostGetFlags(pFlags: *mut ::std::os::raw::c_uint, pHost: *mut ::std::os::raw::c_void) -> cudaError_t;

    pub fn cudaMalloc3D(pitchedDevPtr: *mut cudaPitchedPtr, extent: cudaExtent) -> cudaError_t;

    pub fn cudaMalloc3DArray(
        array: *mut cudaArray_t,
        desc: *const cudaChannelFormatDesc,
        extent: cudaExtent,
        flags: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaMallocMipmappedArray(
        mipmappedArray: *mut cudaMipmappedArray_t,
        desc: *const cudaChannelFormatDesc,
        extent: cudaExtent,
        numLevels: ::std::os::raw::c_uint,
        flags: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaGetMipmappedArrayLevel(
        levelArray: *mut cudaArray_t,
        mipmappedArray: cudaMipmappedArray_const_t,
        level: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaMemcpy3D(p: *const cudaMemcpy3DParms) -> cudaError_t;

    pub fn cudaMemcpy3DPeer(p: *const cudaMemcpy3DPeerParms) -> cudaError_t;

    pub fn cudaMemcpy3DAsync(p: *const cudaMemcpy3DParms, stream: cudaStream_t) -> cudaError_t;

    pub fn cudaMemcpy3DPeerAsync(p: *const cudaMemcpy3DPeerParms, stream: cudaStream_t) -> cudaError_t;

    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;

    pub fn cudaArrayGetInfo(
        desc: *mut cudaChannelFormatDesc,
        extent: *mut cudaExtent,
        flags: *mut ::std::os::raw::c_uint,
        array: cudaArray_t,
    ) -> cudaError_t;

    pub fn cudaMemcpy(
        dst: *mut ::std::os::raw::c_void,
        src: *const ::std::os::raw::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;

    pub fn cudaMemcpyPeer(
        dst: *mut ::std::os::raw::c_void,
        dstDevice: ::std::os::raw::c_int,
        src: *const ::std::os::raw::c_void,
        srcDevice: ::std::os::raw::c_int,
        count: usize,
    ) -> cudaError_t;

    pub fn cudaMemcpyToArray(
        dst: cudaArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const ::std::os::raw::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;

    pub fn cudaMemcpyFromArray(
        dst: *mut ::std::os::raw::c_void,
        src: cudaArray_const_t,
        wOffset: usize,
        hOffset: usize,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;

    pub fn cudaMemcpyArrayToArray(
        dst: cudaArray_t,
        wOffsetDst: usize,
        hOffsetDst: usize,
        src: cudaArray_const_t,
        wOffsetSrc: usize,
        hOffsetSrc: usize,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;

    pub fn cudaMemcpy2D(
        dst: *mut ::std::os::raw::c_void,
        dpitch: usize,
        src: *const ::std::os::raw::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;

    pub fn cudaMemcpy2DToArray(
        dst: cudaArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const ::std::os::raw::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;

    pub fn cudaMemcpy2DFromArray(
        dst: *mut ::std::os::raw::c_void,
        dpitch: usize,
        src: cudaArray_const_t,
        wOffset: usize,
        hOffset: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;

    pub fn cudaMemcpy2DArrayToArray(
        dst: cudaArray_t,
        wOffsetDst: usize,
        hOffsetDst: usize,
        src: cudaArray_const_t,
        wOffsetSrc: usize,
        hOffsetSrc: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;

    pub fn cudaMemcpyToSymbol(
        symbol: *const ::std::os::raw::c_void,
        src: *const ::std::os::raw::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;

    pub fn cudaMemcpyFromSymbol(
        dst: *mut ::std::os::raw::c_void,
        symbol: *const ::std::os::raw::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;

    pub fn cudaMemcpyAsync(
        dst: *mut ::std::os::raw::c_void,
        src: *const ::std::os::raw::c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaMemcpyPeerAsync(
        dst: *mut ::std::os::raw::c_void,
        dstDevice: ::std::os::raw::c_int,
        src: *const ::std::os::raw::c_void,
        srcDevice: ::std::os::raw::c_int,
        count: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaMemcpyToArrayAsync(
        dst: cudaArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const ::std::os::raw::c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaMemcpyFromArrayAsync(
        dst: *mut ::std::os::raw::c_void,
        src: cudaArray_const_t,
        wOffset: usize,
        hOffset: usize,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaMemcpy2DAsync(
        dst: *mut ::std::os::raw::c_void,
        dpitch: usize,
        src: *const ::std::os::raw::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaMemcpy2DToArrayAsync(
        dst: cudaArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const ::std::os::raw::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaMemcpy2DFromArrayAsync(
        dst: *mut ::std::os::raw::c_void,
        dpitch: usize,
        src: cudaArray_const_t,
        wOffset: usize,
        hOffset: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaMemcpyToSymbolAsync(
        symbol: *const ::std::os::raw::c_void,
        src: *const ::std::os::raw::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaMemcpyFromSymbolAsync(
        dst: *mut ::std::os::raw::c_void,
        symbol: *const ::std::os::raw::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaMemset(devPtr: *mut ::std::os::raw::c_void, value: ::std::os::raw::c_int, count: usize) -> cudaError_t;

    pub fn cudaMemset2D(
        devPtr: *mut ::std::os::raw::c_void,
        pitch: usize,
        value: ::std::os::raw::c_int,
        width: usize,
        height: usize,
    ) -> cudaError_t;

    pub fn cudaMemset3D(pitchedDevPtr: cudaPitchedPtr, value: ::std::os::raw::c_int, extent: cudaExtent)
        -> cudaError_t;

    pub fn cudaMemsetAsync(
        devPtr: *mut ::std::os::raw::c_void,
        value: ::std::os::raw::c_int,
        count: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaMemset2DAsync(
        devPtr: *mut ::std::os::raw::c_void,
        pitch: usize,
        value: ::std::os::raw::c_int,
        width: usize,
        height: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaMemset3DAsync(
        pitchedDevPtr: cudaPitchedPtr,
        value: ::std::os::raw::c_int,
        extent: cudaExtent,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaGetSymbolAddress(
        devPtr: *mut *mut ::std::os::raw::c_void,
        symbol: *const ::std::os::raw::c_void,
    ) -> cudaError_t;

    pub fn cudaGetSymbolSize(size: *mut usize, symbol: *const ::std::os::raw::c_void) -> cudaError_t;

    pub fn cudaMemPrefetchAsync(
        devPtr: *const ::std::os::raw::c_void,
        count: usize,
        dstDevice: ::std::os::raw::c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaMemAdvise(
        devPtr: *const ::std::os::raw::c_void,
        count: usize,
        advice: cudaMemoryAdvise,
        device: ::std::os::raw::c_int,
    ) -> cudaError_t;

    pub fn cudaMemRangeGetAttribute(
        data: *mut ::std::os::raw::c_void,
        dataSize: usize,
        attribute: cudaMemRangeAttribute,
        devPtr: *const ::std::os::raw::c_void,
        count: usize,
    ) -> cudaError_t;

    pub fn cudaMemRangeGetAttributes(
        data: *mut *mut ::std::os::raw::c_void,
        dataSizes: *mut usize,
        attributes: *mut cudaMemRangeAttribute,
        numAttributes: usize,
        devPtr: *const ::std::os::raw::c_void,
        count: usize,
    ) -> cudaError_t;

    pub fn cudaPointerGetAttributes(
        attributes: *mut cudaPointerAttributes,
        ptr: *const ::std::os::raw::c_void,
    ) -> cudaError_t;

    pub fn cudaDeviceCanAccessPeer(
        canAccessPeer: *mut ::std::os::raw::c_int,
        device: ::std::os::raw::c_int,
        peerDevice: ::std::os::raw::c_int,
    ) -> cudaError_t;

    pub fn cudaDeviceEnablePeerAccess(peerDevice: ::std::os::raw::c_int, flags: ::std::os::raw::c_uint) -> cudaError_t;

    pub fn cudaDeviceDisablePeerAccess(peerDevice: ::std::os::raw::c_int) -> cudaError_t;

    pub fn cudaGraphicsUnregisterResource(resource: cudaGraphicsResource_t) -> cudaError_t;

    pub fn cudaGraphicsResourceSetMapFlags(
        resource: cudaGraphicsResource_t,
        flags: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaGraphicsMapResources(
        count: ::std::os::raw::c_int,
        resources: *mut cudaGraphicsResource_t,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaGraphicsUnmapResources(
        count: ::std::os::raw::c_int,
        resources: *mut cudaGraphicsResource_t,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaGraphicsResourceGetMappedPointer(
        devPtr: *mut *mut ::std::os::raw::c_void,
        size: *mut usize,
        resource: cudaGraphicsResource_t,
    ) -> cudaError_t;

    pub fn cudaGraphicsSubResourceGetMappedArray(
        array: *mut cudaArray_t,
        resource: cudaGraphicsResource_t,
        arrayIndex: ::std::os::raw::c_uint,
        mipLevel: ::std::os::raw::c_uint,
    ) -> cudaError_t;

    pub fn cudaGraphicsResourceGetMappedMipmappedArray(
        mipmappedArray: *mut cudaMipmappedArray_t,
        resource: cudaGraphicsResource_t,
    ) -> cudaError_t;

    pub fn cudaGetChannelDesc(desc: *mut cudaChannelFormatDesc, array: cudaArray_const_t) -> cudaError_t;

    pub fn cudaCreateChannelDesc(
        x: ::std::os::raw::c_int,
        y: ::std::os::raw::c_int,
        z: ::std::os::raw::c_int,
        w: ::std::os::raw::c_int,
        f: cudaChannelFormatKind,
    ) -> cudaChannelFormatDesc;

    pub fn cudaBindTexture(
        offset: *mut usize,
        texref: *const textureReference,
        devPtr: *const ::std::os::raw::c_void,
        desc: *const cudaChannelFormatDesc,
        size: usize,
    ) -> cudaError_t;

    pub fn cudaBindTexture2D(
        offset: *mut usize,
        texref: *const textureReference,
        devPtr: *const ::std::os::raw::c_void,
        desc: *const cudaChannelFormatDesc,
        width: usize,
        height: usize,
        pitch: usize,
    ) -> cudaError_t;

    pub fn cudaBindTextureToArray(
        texref: *const textureReference,
        array: cudaArray_const_t,
        desc: *const cudaChannelFormatDesc,
    ) -> cudaError_t;

    pub fn cudaBindTextureToMipmappedArray(
        texref: *const textureReference,
        mipmappedArray: cudaMipmappedArray_const_t,
        desc: *const cudaChannelFormatDesc,
    ) -> cudaError_t;

    pub fn cudaUnbindTexture(texref: *const textureReference) -> cudaError_t;

    pub fn cudaGetTextureAlignmentOffset(offset: *mut usize, texref: *const textureReference) -> cudaError_t;

    pub fn cudaGetTextureReference(
        texref: *mut *const textureReference,
        symbol: *const ::std::os::raw::c_void,
    ) -> cudaError_t;

    pub fn cudaBindSurfaceToArray(
        surfref: *const surfaceReference,
        array: cudaArray_const_t,
        desc: *const cudaChannelFormatDesc,
    ) -> cudaError_t;

    pub fn cudaGetSurfaceReference(
        surfref: *mut *const surfaceReference,
        symbol: *const ::std::os::raw::c_void,
    ) -> cudaError_t;

    pub fn cudaCreateTextureObject(
        pTexObject: *mut cudaTextureObject_t,
        pResDesc: *const cudaResourceDesc,
        pTexDesc: *const cudaTextureDesc,
        pResViewDesc: *const cudaResourceViewDesc,
    ) -> cudaError_t;

    pub fn cudaDestroyTextureObject(texObject: cudaTextureObject_t) -> cudaError_t;

    pub fn cudaGetTextureObjectResourceDesc(
        pResDesc: *mut cudaResourceDesc,
        texObject: cudaTextureObject_t,
    ) -> cudaError_t;

    pub fn cudaGetTextureObjectTextureDesc(
        pTexDesc: *mut cudaTextureDesc,
        texObject: cudaTextureObject_t,
    ) -> cudaError_t;

    pub fn cudaGetTextureObjectResourceViewDesc(
        pResViewDesc: *mut cudaResourceViewDesc,
        texObject: cudaTextureObject_t,
    ) -> cudaError_t;

    pub fn cudaCreateSurfaceObject(
        pSurfObject: *mut cudaSurfaceObject_t,
        pResDesc: *const cudaResourceDesc,
    ) -> cudaError_t;

    pub fn cudaDestroySurfaceObject(surfObject: cudaSurfaceObject_t) -> cudaError_t;

    pub fn cudaGetSurfaceObjectResourceDesc(
        pResDesc: *mut cudaResourceDesc,
        surfObject: cudaSurfaceObject_t,
    ) -> cudaError_t;

    pub fn cudaDriverGetVersion(driverVersion: *mut ::std::os::raw::c_int) -> cudaError_t;

    pub fn cudaRuntimeGetVersion(runtimeVersion: *mut ::std::os::raw::c_int) -> cudaError_t;

    pub fn cudaGetExportTable(
        ppExportTable: *mut *const ::std::os::raw::c_void,
        pExportTableId: *const cudaUUID_t,
    ) -> cudaError_t;
}
