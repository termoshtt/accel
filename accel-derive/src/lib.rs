#![feature(proc_macro)]
#![recursion_limit = "128"]

extern crate nvptx;
extern crate proc_macro;
#[macro_use]
extern crate quote;
extern crate syn;

mod generate;
mod parse;

use proc_macro::TokenStream;
use syn::{FnArg, ItemFn};
use nvptx::{parse_func, func2kernel};

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, func: TokenStream) -> TokenStream {
    let func = parse_func(func);
    let ptx_str = func2kernel(&func);
    func2caller(&ptx_str, &func)
}

fn func2caller(ptx_str: &str, func: &ItemFn) -> TokenStream {
    let vis = &func.vis;
    let ident = &func.ident;

    let decl = &func.decl;
    let inputs = &decl.inputs;
    let output = &decl.output;
    let fn_token = &decl.fn_token;

    let input_values: Vec<_> = inputs
        .iter()
        .map(|arg| match arg {
            &FnArg::Captured(ref val) => &val.pat,
            _ => unreachable!(""),
        })
        .collect();
    let kernel_name = quote!{ #ident }.to_string();

    let caller = quote!{
        mod ptx_mod {
            use ::std::cell::RefCell;
            use ::accel::module::Module;
            thread_local! {
                #[allow(non_upper_case_globals)]
                pub static #ident: RefCell<Module>
                    = RefCell::new(Module::from_str(#ptx_str).expect("Load module failed"));
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
