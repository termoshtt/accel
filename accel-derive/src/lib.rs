//! proc_macro for accel's #[kernel]
#![recursion_limit = "128"]

extern crate nvptx;
extern crate proc_macro;
#[macro_use]
extern crate quote;
extern crate syn;

mod generate;
mod parse;

use proc_macro::TokenStream;

use generate::func2kernel;
use parse::parse_func;

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, func: TokenStream) -> TokenStream {
    let func = parse_func(func);
    let ptx_str = func2kernel(&func);
    func2caller(&ptx_str, &func)
}
