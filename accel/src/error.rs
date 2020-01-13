pub use cuda::cudaError_enum as DeviceError;
pub use cudart::cudaError_t as RuntimeError;

pub type Result<T> = ::std::result::Result<T, AccelError>;

#[derive(thiserror::Error, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AccelError {
    #[error("CUDA Device API Error: {0:?}")]
    Device(DeviceError),
    #[error("CUDA Runtime API Error: {0:?}")]
    Runtime(RuntimeError),
}

/// Convert return code of CUDA Driver/Runtime API into Result
pub(crate) trait Check {
    fn check(self) -> Result<()>;
}

impl Check for DeviceError {
    fn check(self) -> Result<()> {
        if self == DeviceError::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(self.into())
        }
    }
}

impl Check for RuntimeError {
    fn check(self) -> Result<()> {
        if self == RuntimeError::cudaSuccess {
            Ok(())
        } else {
            Err(self.into())
        }
    }
}

impl Into<AccelError> for DeviceError {
    fn into(self) -> AccelError {
        AccelError::Device(self)
    }
}

impl Into<AccelError> for RuntimeError {
    fn into(self) -> AccelError {
        AccelError::Runtime(self)
    }
}
