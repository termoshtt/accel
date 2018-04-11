//! Submodule for CUDA device functions
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]
#![allow(dead_code)]

#[repr(u32)]
#[derive(Debug, PartialEq, Eq)]
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

impl cudaError_t {
    pub fn check(self) {
        if self != cudaError_t::Success {
            panic!("Error Code = {}", self as u32);
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
pub type cudaStream_t = *mut CUstream_st;

#[repr(C)]
#[derive(Debug)]
pub struct CUevent_st {
    _unused: [u8; 0],
}
pub type cudaEvent_t = *mut CUevent_st;

bitflags! {
    #[repr(C)]
    pub struct cudaStreamFlags: u32 {
        const Default = 0b00000000;
        const NonBlocking = 0b00000001;
    }
}

bitflags! {
    #[repr(C)]
    pub struct cudaEventFlags: u32 {
        const Default = 0b00000000;
        const BlockingSync= 0b00000001;
        const DisableTiming= 0b00000010;
        const Interprocess= 0b00000100;
    }
}

use libc::c_void;
use super::Dim3;

extern "C" {

    // device steram APIs
    pub fn cudaStreamCreateWithFlags(pStream: *mut cudaStream_t, flags: cudaStreamFlags) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamWaitEvent(
        stream: cudaStream_t,
        event: cudaEvent_t,
        flags: u32, /* must be zero */
    ) -> cudaError_t;

    // device event APIs
    pub fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: cudaEventFlags) -> cudaError_t;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;

    // launch from device
    pub fn cudaGetParameterBufferV2(func: *mut c_void, grid: Dim3, block: Dim3, sharedMemSize: u32) -> *mut c_void;
    pub fn cudaLaunchDeviceV2(parameterBuffer: *mut c_void, stream: cudaStream_t) -> cudaError_t;
}
