pub use cuda::cudaError_enum as DeviceError;
pub use cudart::cudaError_t as RuntimeError;

pub type Result<T> = ::std::result::Result<T, Error>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Error {
    Device(DeviceError),
    Runtime(RuntimeError),
}

pub trait Check {
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

impl Into<Error> for DeviceError {
    fn into(self) -> Error {
        Error::Device(self)
    }
}

impl Into<Error> for RuntimeError {
    fn into(self) -> Error {
        Error::Runtime(self)
    }
}
