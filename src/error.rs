#![allow(non_camel_case_types)]

use ffi::cuda_runtime as rt;
pub use ffi::cuda::cudaError_enum as cudaError;

pub type Result<T> = ::std::result::Result<T, Error>;

#[macro_export]
macro_rules! cudo {
    ($st:expr) => {
        $crate::error::check(unsafe { $st })?;
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, IntoEnum)]
pub enum Error {
    cudaError(cudaError),
    cudaRuntimeError(cudaRuntimeError),
}

pub fn check<E: Into<Error>>(e: E) -> Result<()> {
    match e.into() {
        Error::cudaError(e) => {
            if e != cudaError::CUDA_SUCCESS {
                return Err(e.into());
            }
        }
        Error::cudaRuntimeError(e) => {
            if e != cudaRuntimeError::Success {
                return Err(e.into());
            }
        }
    }
    Ok(())
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum cudaRuntimeError {
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
impl From<rt::cudaError> for Error {
    fn from(e: rt::cudaError) -> Error {
        let e: cudaRuntimeError = e.into();
        e.into()
    }
}

impl From<rt::cudaError> for cudaRuntimeError {
    fn from(e: rt::cudaError) -> Self {
        use self::cudaRuntimeError::*;
        match e {
            0 => Success,
            1 => MissingConfiguration,
            2 => MemoryAllocation,
            3 => InitializationError,
            4 => LaunchFailure,
            5 => PriorLaunchFailure,
            6 => LaunchTimeout,
            7 => LaunchOutOfResources,
            8 => InvalidDeviceFunction,
            9 => InvalidConfiguration,
            10 => InvalidDevice,
            11 => InvalidValue,
            12 => InvalidPitchValue,
            13 => InvalidSymbol,
            14 => MapBufferObjectFailed,
            15 => UnmapBufferObjectFailed,
            16 => InvalidHostPointer,
            17 => InvalidDevicePointer,
            18 => InvalidTexture,
            19 => InvalidTextureBinding,
            20 => InvalidChannelDescriptor,
            21 => InvalidMemcpyDirection,
            22 => AddressOfConstant,
            23 => TextureFetchFailed,
            24 => TextureNotBound,
            25 => SynchronizationError,
            26 => InvalidFilterSetting,
            27 => InvalidNormSetting,
            28 => MixedDeviceExecution,
            29 => CudartUnloading,
            30 => Unknown,
            31 => NotYetImplemented,
            32 => MemoryValueTooLarge,
            33 => InvalidResourceHandle,
            34 => NotReady,
            35 => InsufficientDriver,
            36 => SetOnActiveProcess,
            37 => InvalidSurface,
            38 => NoDevice,
            39 => ECCUncorrectable,
            40 => SharedObjectSymbolNotFound,
            41 => SharedObjectInitFailed,
            42 => UnsupportedLimit,
            43 => DuplicateVariableName,
            44 => DuplicateTextureName,
            45 => DuplicateSurfaceName,
            46 => DevicesUnavailable,
            47 => InvalidKernelImage,
            48 => NoKernelImageForDevice,
            49 => IncompatibleDriverContext,
            50 => PeerAccessAlreadyEnabled,
            51 => PeerAccessNotEnabled,
            54 => DeviceAlreadyInUse,
            55 => ProfilerDisabled,
            56 => ProfilerNotInitialized,
            57 => ProfilerAlreadyStarted,
            58 => ProfilerAlreadyStopped,
            59 => Assert,
            60 => TooManyPeers,
            61 => HostMemoryAlreadyRegistered,
            62 => HostMemoryNotRegistered,
            63 => OperatingSystem,
            64 => PeerAccessUnsupported,
            65 => LaunchMaxDepthExceeded,
            66 => LaunchFileScopedTex,
            67 => LaunchFileScopedSurf,
            68 => SyncDepthExceeded,
            69 => LaunchPendingCountExceeded,
            70 => NotPermitted,
            71 => NotSupported,
            72 => HardwareStackError,
            73 => IllegalInstruction,
            74 => MisalignedAddress,
            75 => InvalidAddressSpace,
            76 => InvalidPc,
            77 => IllegalAddress,
            78 => InvalidPtx,
            79 => InvalidGraphicsContext,
            80 => NvlinkUncorrectable,
            81 => JitCompilerNotFound,
            82 => CooperativeLaunchTooLarge,
            127 => StartupFailure,
            10000 => ApiFailureBase,
            _ => unreachable!("Invalid return value"),
        }
    }
}
