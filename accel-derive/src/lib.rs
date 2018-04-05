#![feature(proc_macro)]
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

mod config;
mod build;
mod parse;
mod generate;

use proc_macro::TokenStream;

use parse::*;
use generate::*;

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, func: TokenStream) -> TokenStream {
    let func = Function::parse(func);
    let ptx_str = func2kernel(&func);
    func2caller(&ptx_str, &func)
}
