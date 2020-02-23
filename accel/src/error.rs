pub use cuda::cudaError_enum as DeviceError;
pub use cudart::cudaError_t as RuntimeError;

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AccelError {
    #[error("CUDA Device API Error: {api_name}, {error:?}")]
    Device {
        api_name: String,
        error: DeviceError,
    },

    #[error("CUDA Runtime API Error: {api_name}, {error:?}")]
    Runtime {
        api_name: String,
        error: RuntimeError,
    },
}

/// Convert return code of CUDA Driver/Runtime API into Result
pub(crate) trait Check {
    fn check(self, api_name: &str) -> Result<(), AccelError>;
}

impl Check for DeviceError {
    fn check(self, api_name: &str) -> Result<(), AccelError> {
        if self == DeviceError::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(AccelError::Device {
                api_name: api_name.into(),
                error: self,
            })
        }
    }
}

impl Check for RuntimeError {
    fn check(self, api_name: &str) -> Result<(), AccelError> {
        if self == RuntimeError::cudaSuccess {
            Ok(())
        } else {
            Err(AccelError::Runtime {
                api_name: api_name.into(),
                error: self,
            })
        }
    }
}
