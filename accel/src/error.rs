pub use cuda::cudaError_enum as DeviceError;
use std::path::PathBuf;

pub type Result<T> = ::std::result::Result<T, AccelError>;

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AccelError {
    /// Raw errors originates from CUDA Device APIs
    #[error("CUDA Device API Error: {api_name}, {error:?}")]
    Device {
        api_name: String,
        error: DeviceError,
    },

    // This is not an error potentially, but it should be a bug if not captured by accel
    #[error("Async operations issues previously have not completed yet")]
    AsyncOperationNotReady,

    #[error("Current CUDA context does not equal to the context when the object is generated")]
    ContextIsNotCurrent,

    #[error("Context already exists on this thread. Please pop it before push new context.")]
    ContextDuplicated,

    #[error("Given device memory cannot be accessed from CPU because it is not a managed memory")]
    DeviceMemoryIsNotManaged,

    #[error("File not found: {path:?}")]
    FileNotFound { path: PathBuf },
}

/// Convert return code of CUDA Driver/Runtime API into Result
pub(crate) trait Check {
    fn check(self, api_name: &str) -> Result<()>;
}

impl Check for DeviceError {
    fn check(self, api_name: &str) -> Result<()> {
        match self {
            DeviceError::CUDA_SUCCESS => Ok(()),
            DeviceError::CUDA_ERROR_NOT_READY => Err(AccelError::AsyncOperationNotReady),
            _ => Err(AccelError::Device {
                api_name: api_name.into(),
                error: self,
            }),
        }
    }
}

#[macro_export]
macro_rules! ffi_call {
    ($ffi:path, $($args:expr),*) => {
        $ffi($($args),*).check(stringify!($ffi))
    };
    ($ffi:path) => {
        $ffi().check(stringify!($ffi))
    };
}

#[macro_export]
macro_rules! ffi_call_unsafe {
    ($ffi:path, $($args:expr),*) => {
        unsafe { $crate::error::Check::check($ffi($($args),*), stringify!($ffi)) }
    };
    ($ffi:path) => {
        unsafe { $crate::error::Check::check($ffi(), stringify!($ffi)) }
    };
}

#[macro_export]
macro_rules! ffi_new {
    ($ffi:path, $($args:expr),*) => {
        {
            let mut value = ::std::mem::MaybeUninit::uninit();
            $crate::error::Check::check($ffi(value.as_mut_ptr(), $($args),*), stringify!($ffi)).map(|_| value.assume_init())
        }
    };
    ($ffi:path) => {
        {
            let mut value = ::std::mem::MaybeUninit::uninit();
            $crate::error::Check::check($ffi(value.as_mut_ptr()), stringify!($ffi)).map(|_| value.assume_init())
        }
    };
}

#[macro_export]
macro_rules! ffi_new_unsafe {
    ($ffi:path, $($args:expr),*) => {
        unsafe { $crate::ffi_new!($ffi, $($args),*) }
    };
    ($ffi:path) => {
        unsafe { $crate::ffi_new!($ffi) }
    };
}
