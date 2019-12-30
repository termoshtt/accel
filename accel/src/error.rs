#![allow(non_camel_case_types)]

pub use ffi::cublas::cublasStatus_t as cublasError;
pub use ffi::cuda::cudaError_t as cudaError;
pub use ffi::cudart::cudaError_t as cudaRuntimeError;

pub type Result<T> = ::std::result::Result<T, Error>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, IntoEnum)]
pub enum Error {
    cudaError(cudaError),
    cudaRuntimeError(cudaRuntimeError),
    cublasError(cublasError),
}

pub trait Check {
    fn check(self) -> Result<()>;
}

impl Check for cudaError {
    fn check(self) -> Result<()> {
        if self == cudaError::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(self.into())
        }
    }
}

impl Check for cudaRuntimeError {
    fn check(self) -> Result<()> {
        if self == cudaRuntimeError::Success {
            Ok(())
        } else {
            Err(self.into())
        }
    }
}

impl Check for cublasError {
    fn check(self) -> Result<()> {
        if self == cublasError::SUCCESS {
            Ok(())
        } else {
            Err(self.into())
        }
    }
}
