#![allow(non_camel_case_types)]

pub use cuda::cudaError_enum as cudaError;
pub use cudart::cudaError_t as cudaRuntimeError;

pub type Result<T> = ::std::result::Result<T, Error>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Error {
    cudaError(cudaError),
    cudaRuntimeError(cudaRuntimeError),
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
        if self == cudaRuntimeError::cudaSuccess {
            Ok(())
        } else {
            Err(self.into())
        }
    }
}

impl Into<Error> for cudaError {
    fn into(self) -> Error {
        Error::cudaError(self)
    }
}

impl Into<Error> for cudaRuntimeError {
    fn into(self) -> Error {
        Error::cudaRuntimeError(self)
    }
}
