#![feature(proc_macro)]
#![recursion_limit = "128"]

extern crate glob;
extern crate proc_macro;
#[macro_use]
extern crate procedurals;
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

use proc_macro::TokenStream;
use syn::*;

use parse::*;
use build::*;

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, func: TokenStream) -> TokenStream {
    let func = Function::parse(func);
    let ptx_str = func2kernel(&func);
    func2caller(&ptx_str, &func)
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

    let attrs = parse_attrs(func);
    let mut builder = Builder::new(attrs);

    let crates: Vec<Ident> = builder
        .depends
        .iter()
        .map(|c| c.name().replace("-", "_").into())
        .collect();
    let kernel_str = quote!{
        #![feature(abi_ptx)]
        #![no_std]
        #(extern crate #crates;), *
        #[no_mangle]
        #vis #unsafety extern "ptx-kernel" #fn_token #ident(#inputs) #output #block
    }.to_string();
    builder.compile(&kernel_str)
}

fn func2caller(ptx_str: &str, func: &Function) -> TokenStream {
    let vis = &func.vis;
    let fn_token = &func.fn_token;
    let ident = &func.ident;
    let inputs = &func.inputs;
    let output = &func.output;

    let input_values = func.input_values();
    let kernel_name = quote!{ #ident }.to_string();

    let caller = quote!{
        mod ptx_mod {
            use ::std::cell::RefCell;
            use ::accel::module::Module;
            thread_local! {
                #[allow(non_upper_case_globals)]
                pub static #ident: RefCell<Module> = RefCell::new(Module::from_str(#ptx_str).expect("Load module failed"));
            }
        }
        #vis #fn_token #ident(grid: ::accel::Grid, block: ::accel::Block, #inputs) #output {
            use ::accel::kernel::void_cast;
            ptx_mod::#ident.with(|m| {
                let m = m.borrow();
                let mut kernel = m.get_kernel(#kernel_name).expect("Failed to get Kernel");
                let mut args = [#(void_cast(&#input_values)),*];
                unsafe { kernel.launch(args.as_mut_ptr(), grid, block).expect("Failed to launch kernel") };
            })
        }
    };
    caller.into()
}
