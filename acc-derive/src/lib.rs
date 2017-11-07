#![feature(proc_macro)]
#![recursion_limit = "128"]

#[macro_use]
extern crate futures_await_quote as quote;
extern crate futures_await_syn as syn;
extern crate proc_macro;

extern crate acc;

use proc_macro::TokenStream;
use syn::*;

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, func: TokenStream) -> TokenStream {
    let ptx_kernel = func2kernel(&func);
    let ptx = acc::ptx_builder::compile(&ptx_kernel.to_string());
    println!("PTX = {}", ptx);
    func2caller(&func)
}

/// Convert function decorated by #[kernel] into a single `lib.rs` for PTX-builder
fn func2kernel(func: &TokenStream) -> TokenStream {
    let Item { node, .. } = syn::parse(func.clone()).unwrap();
    let ItemFn {
        ident,
        vis,
        block,
        decl,
        ..
    } = match node {
        ItemKind::Fn(item) => item,
        _ => unreachable!(""),
    };

    let FnDecl {
        inputs,
        output,
        fn_token,
        ..
    } = {
        *decl
    };

    let kernel =
        quote!{
        #![feature(abi_ptx)]
        #![no_std]
        #[no_mangle]
        #vis extern "ptx-kernel" #fn_token #ident(#inputs) #output #block
    };
    kernel.into()
}

fn func2caller(func: &TokenStream) -> TokenStream {
    let Item { node, .. } = syn::parse(func.clone()).unwrap();
    let ItemFn {
        ident,
        vis,
        decl,
        ..  // for future compatiblity
    } = match node {
        ItemKind::Fn(item) => item,
        _ => unreachable!("")
    };

    let FnDecl {
        inputs,
        output,
        fn_token,
        ..
    } = {
        *decl
    };

    // FIXME call kernel
    let caller =
        quote!{
        #vis #fn_token #ident(#inputs) #output {}
    };
    caller.into()
}
