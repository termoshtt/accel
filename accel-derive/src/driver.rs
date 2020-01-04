use crate::parser::*;
use failure::*;

pub struct Driver {}

impl Driver {
    pub fn from_attrs(_attrs: &Attributes) -> Fallible<Self> {
        unimplemented!()
    }

    pub fn compile_str(&self, _rust_str: &str) -> Fallible<String> {
        unimplemented!()
    }
}
