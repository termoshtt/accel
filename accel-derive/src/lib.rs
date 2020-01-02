#![recursion_limit = "128"]

extern crate proc_macro;

mod device;
mod host;
mod parser;

use proc_macro::TokenStream;

use device::*;
use host::*;

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, func: TokenStream) -> TokenStream {
    let func = syn::parse(func).expect("Not a function");
    let ptx_str = func2kernel(&func);
    func2caller(&ptx_str, &func)
}
