#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

pub use driver_types::*;
pub use library_types::*;
use vector_types::*;

include!(concat!(env!("OUT_DIR"), "/cudart_bindings.rs"));
