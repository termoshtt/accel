use cuda::cudaError_enum as DeviceError;
use std::path::PathBuf;

pub type Result<T> = ::std::result::Result<T, AccelError>;

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AccelError {
    /// Raw errors originates from CUDA Device APIs
    #[error("CUDA Device API Error: {api_name}, {error:?}")]
    CUDAError {
        api_name: String,
        error: DeviceError,
    },

    // This is not an error potentially, but it should be a bug if not captured by accel
    #[error("Async operations issues previously have not completed yet")]
    AsyncOperationNotReady,

    /// Error for user device code assertion
    #[error("Assertion in device code has failed")]
    DeviceAssertionFailed,

    #[error("No device found for given ID")]
    DeviceNotFound { id: usize, count: usize },

    #[error("File not found: {path:?}")]
    FileNotFound { path: PathBuf },
}

/// Convert return code of CUDA Driver/Runtime API into Result
pub(crate) fn check(error: DeviceError, api_name: &str) -> Result<()> {
    match error {
        DeviceError::CUDA_SUCCESS => Ok(()),
        DeviceError::CUDA_ERROR_ASSERT => Err(AccelError::DeviceAssertionFailed),
        DeviceError::CUDA_ERROR_NOT_READY => Err(AccelError::AsyncOperationNotReady),
        _ => Err(AccelError::CUDAError {
            api_name: api_name.into(),
            error,
        }),
    }
}

#[macro_export]
macro_rules! ffi_call {
    ($ffi:path $(,$args:expr)*) => {
        unsafe {
            $crate::error::check($ffi($($args),*), stringify!($ffi))
        }
    };
}

#[macro_export]
macro_rules! ffi_new {
    ($ffi:path $(,$args:expr)*) => {
        unsafe {
            let mut value = ::std::mem::MaybeUninit::uninit();
            $crate::error::check($ffi(value.as_mut_ptr(), $($args),*), stringify!($ffi)).map(|_| value.assume_init())
        }
    };
}
