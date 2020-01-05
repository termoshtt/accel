#![recursion_limit = "128"]

extern crate proc_macro;

mod builder;
mod host;
mod parser;

use proc_macro::TokenStream;

use host::*;

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, func: TokenStream) -> TokenStream {
    let func: syn::ItemFn = syn::parse(func).expect("Not a function");
    let ptx_str = builder::compile_tokens(&func).expect("Failed to compile to PTX");
    func2caller(&ptx_str, &func)
}
