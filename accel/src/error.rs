pub use cuda::cudaError_enum as DeviceError;
pub use cudart::cudaError_t as RuntimeError;

pub type Result<T> = ::std::result::Result<T, AccelError>;

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AccelError {
    /// Raw errors originates from CUDA Device APIs
    #[error("CUDA Device API Error: {api_name}, {error:?}")]
    Device {
        api_name: String,
        error: DeviceError,
    },

    /// Raw errors originates from CUDA Device APIs
    #[error("CUDA Runtime API Error: {api_name}, {error:?}")]
    Runtime {
        api_name: String,
        error: RuntimeError,
    },
}

/// Convert return code of CUDA Driver/Runtime API into Result
pub(crate) trait Check {
    fn check(self, api_name: &str) -> Result<()>;
}

impl Check for DeviceError {
    fn check(self, api_name: &str) -> Result<()> {
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
    fn check(self, api_name: &str) -> Result<()> {
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
