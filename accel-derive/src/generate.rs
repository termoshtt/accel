//! Generate Rust code

use proc_macro::TokenStream;
use syn::{FnArg, Ident, ItemFn};

use parse::parse_builder_attrs;

fn header(crates: &[String]) -> String {
    let crates: Vec<Ident> = crates
        .iter()
        .map(|c| Ident::from(c.replace("-", "_")))
        .collect();
    let tt = quote!{
        #![feature(abi_ptx)]
        #![no_std]
        #(extern crate #crates;), *
    };
    tt.to_string()
}

fn kernel(func: &ItemFn) -> String {
    let vis = &func.vis;
    let ident = &func.ident;
    let unsafety = &func.unsafety;
    let block = &func.block;

    let decl = &func.decl;
    let fn_token = &decl.fn_token;
    let inputs = &decl.inputs;
    let output = &decl.output;

    let kernel = quote!{
        #[no_mangle]
        #vis #unsafety extern "ptx-kernel" #fn_token #ident(#inputs) #output #block
    };
    kernel.to_string()
}

/// Convert function decorated by #[kernel] into a single `lib.rs` for PTX-builder
fn func2kernel(func: &ItemFn) -> String {
    let mut builder = parse_builder_attrs(&func.attrs);
    let lib = format!("{}\n{}", header(&builder.crates()), kernel(func));
    builder.compile(&lib).expect("Failed to compile")
}

pub fn func2caller(ptx_str: &str, func: &ItemFn) -> TokenStream {
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
