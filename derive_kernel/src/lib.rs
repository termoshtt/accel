#![feature(proc_macro)]

// extern crate proc_macro2;
extern crate proc_macro;
use proc_macro::*;

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, func: TokenStream) -> TokenStream {
    func
}
