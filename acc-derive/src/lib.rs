
#![feature(proc_macro)]
#![recursion_limit = "128"]

extern crate proc_macro2;
extern crate proc_macro;
#[macro_use]
extern crate futures_await_quote as quote;
extern crate futures_await_syn as syn;
extern crate synom;

use proc_macro::TokenStream;
use syn::*;

#[proc_macro_attribute]
pub fn kernel(kernel_attr: TokenStream, function: TokenStream) -> TokenStream {
    let Item { attrs, node } = syn::parse(function).unwrap();
    println!("{:?}", kernel_attr.to_string());
    let ItemFn {
        ident,
        vis,
        unsafety,
        constness,
        abi,
        block,
        decl,
        ..
    } = match node {
        ItemKind::Fn(item) => item,
        _ => panic!("#[kernel] can only be applied to functions"),
    };
    let FnDecl {
        inputs,
        output,
        variadic,
        generics,
        fn_token,
        ..
    } = {
        *decl
    };
    let where_clause = &generics.where_clause;
    let output = quote!{};

    println!(
        "{}",
        quote!{ 
        inputs = #inputs;
        block = #block;
    }
    );
    output.into()
}
