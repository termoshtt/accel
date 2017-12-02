#![feature(proc_macro)]
#![recursion_limit = "128"]

#[macro_use]
extern crate futures_await_quote as quote;
extern crate futures_await_syn as syn;
extern crate proc_macro;

extern crate accel;

use proc_macro::TokenStream;
use syn::*;
use accel::ptx_builder::*;

#[derive(Debug)]
struct Function {
    attrs: Vec<syn::Attribute>,
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
        let Item { node, attrs } = syn::parse(func.clone()).unwrap();
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
            attrs,
            ident,
            vis,
            block,
            unsafety,
            inputs,
            output,
            fn_token,
        }
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
    let ptx_str = func2kernel(&func);
    func2caller(&ptx_str, &func)
}

fn parse_depends(func: &Function) -> Depends {
    let mut deps = Depends::new();
    for attr in &func.attrs {
        let path = &attr.path;
        let path = &quote!{#path}.to_string();
        if path != "depends" {
            unreachable!("Unsupported attribute: {:?}", path);
        }
        let tts = &attr.tts[0];
        let tts = &quote!{#tts}.to_string();
        deps.parse_append(tts);
    }
    deps
}

/// Convert function decorated by #[kernel] into a single `lib.rs` for PTX-builder
fn func2kernel(func: &Function) -> String {
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
    let deps = parse_depends(func);
    compile(&kernel.to_string(), deps)
}

fn func2caller(ptx_str: &str, func: &Function) -> TokenStream {
    let vis = &func.vis;
    let fn_token = &func.fn_token;
    let ident = &func.ident;
    let inputs = &func.inputs;
    let output = &func.output;

    let input_values = func.input_values();
    let kernel_name = quote!{ #ident }.to_string();

    let caller =
        quote!{
        mod ptx_mod {
            use std::cell::RefCell;
            use accel::kernel::PTXModule;
            thread_local! {
                #[allow(non_upper_case_globals)]
                pub static #ident: RefCell<PTXModule> = RefCell::new(PTXModule::from_str(#ptx_str).unwrap());
            }
        }
        #vis #fn_token #ident(grid: ::accel::Grid, block: ::accel::Block, #inputs) #output {
            use accel::kernel::void_cast;
            ptx_mod::#ident.with(|m| {
                let m = m.borrow();
                let mut kernel = m.get_kernel(#kernel_name).unwrap();
                let mut args = [#(void_cast(&#input_values)),*];
                unsafe { kernel.launch(args.as_mut_ptr(), grid, block).unwrap() };
            })
        }
    };
    caller.into()
}
