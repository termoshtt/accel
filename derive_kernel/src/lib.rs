
#![feature(proc_macro)]
#![recursion_limit = "128"]

extern crate proc_macro2;
extern crate proc_macro;
#[macro_use]
extern crate futures_await_quote as quote;
extern crate futures_await_syn as syn;
#[macro_use]
extern crate synom;

use proc_macro2::Span;
use proc_macro::TokenStream;
use quote::{Tokens, ToTokens};
use syn::*;
use syn::delimited::Delimited;
use syn::fold::Folder;

#[proc_macro_attribute]
pub fn kernel(_attribute: TokenStream, function: TokenStream) -> TokenStream {
    let Item { attrs, node } = syn::parse(function).unwrap();
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
    assert!(!variadic, "variadic functions cannot be kernel");
    let (output, rarrow_token) = match output {
        FunctionRetTy::Ty(t, rarrow_token) => (t, rarrow_token),
        FunctionRetTy::Default => {
            (
                TyTup {
                    tys: Default::default(),
                    lone_comma: Default::default(),
                    paren_token: Default::default(),
                }.into(),
                Default::default(),
            )
        }
    };

    let output = quote!{};

    // println!("{}", output);
    output.into()
}
