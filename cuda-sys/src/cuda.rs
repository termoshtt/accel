#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

pub use driver_types::cudaError_t;

include!(concat!(env!("OUT_DIR"), "/cuda_bindings.rs"));
