#![recursion_limit = "128"]

extern crate glob;
extern crate proc_macro;
#[macro_use]
extern crate procedurals;
#[macro_use]
extern crate derive_more;
#[macro_use]
extern crate quote;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate syn;
extern crate tempdir;
extern crate toml;

pub mod config;
pub mod compile;
pub mod parse;
pub mod generate;
