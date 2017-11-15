#![feature(proc_macro)]
#![recursion_limit = "128"]

#[macro_use]
extern crate futures_await_quote as quote;
extern crate futures_await_syn as syn;
extern crate proc_macro;

extern crate accel;

use proc_macro::TokenStream;
use syn::*;
use std::io::Write;

struct Function {
    ident: Ident,
    vis: Visibility,
    block: Box<Block>,
    unsafety: Unsafety,
    inputs: delimited::Delimited<FnArg, tokens::Comma>,
    output: FunctionRetTy,
    fn_token: tokens::Fn_,
}

impl Function {
    fn parse(func: TokenStream) -> Self {
        let Item { node, .. } = syn::parse(func.clone()).unwrap();
        let ItemFn {
            ident,
            vis,
            block,
            decl,
            unsafety,
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
        Function {
            ident,
            vis,
            block,
            unsafety,
            inputs,
            output,
            fn_token,
        }
    }

    fn path(&self) -> String {
        format!("{}_ptx.s", self.ident.to_string())
    }

    fn input_values(&self) -> Vec<&Pat> {
        self.inputs
            .iter()
            .map(|arg| match arg.into_item() {
                &FnArg::Captured(ref val) => &val.pat,
                _ => unreachable!(""),
            })
            .collect()
    }
}

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, func: TokenStream) -> TokenStream {
    let func = Function::parse(func);
    func2kernel(&func);
    func2caller(&func)
}

/// Convert function decorated by #[kernel] into a single `lib.rs` for PTX-builder
fn func2kernel(func: &Function) {
    let vis = &func.vis;
    let fn_token = &func.fn_token;
    let ident = &func.ident;
    let unsafety = &func.unsafety;
    let inputs = &func.inputs;
    let output = &func.output;
    let block = &func.block;

    let kernel =
        quote!{
        #![feature(abi_ptx)]
        #![no_std]
        extern crate accel_core;
        #[no_mangle]
        #vis #unsafety extern "ptx-kernel" #fn_token #ident(#inputs) #output #block
    };

    let ptx = accel::ptx_builder::compile(&kernel.to_string());
    let mut f = ::std::fs::File::create(&func.path()).unwrap();
    f.write(ptx.as_bytes()).unwrap();
}

fn func2caller(func: &Function) -> TokenStream {
    let vis = &func.vis;
    let fn_token = &func.fn_token;
    let ident = &func.ident;
    let inputs = &func.inputs;
    let output = &func.output;

    let input_values = func.input_values();
    let filename = func.path();
    let kernel_name = quote!{ #ident }.to_string();

    let caller =
        quote!{
        #vis #fn_token #ident(grid: ::accel::Grid, block: ::accel::Block, #inputs) #output {
            let ptx = ::accel::kernel::PTXModule::load(#filename).unwrap();
            let mut kernel = ptx.get_function(#kernel_name).unwrap();
            use accel::kernel::void_cast;
            let mut args = [#(void_cast(&#input_values)),*];
            unsafe { kernel.launch(args.as_mut_ptr(), grid, block).unwrap() };
        }
    };
    caller.into()
}
