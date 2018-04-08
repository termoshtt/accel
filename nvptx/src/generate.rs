use proc_macro::TokenStream;

use parse::*;
use compile::*;

pub fn header(builder: &Builder) -> String {
    let crates = builder.crates_for_extern();
    let tt = quote!{
        #![feature(abi_ptx)]
        #![no_std]
        #(extern crate #crates;), *
    };
    tt.to_string()
}

pub fn kernel(func: &Function) -> String {
    let vis = &func.vis;
    let fn_token = &func.fn_token;
    let ident = &func.ident;
    let unsafety = &func.unsafety;
    let inputs = &func.inputs;
    let output = &func.output;
    let block = &func.block;

    let kernel = quote!{
        #[no_mangle]
        #vis #unsafety extern "ptx-kernel" #fn_token #ident(#inputs) #output #block
    };
    kernel.to_string()
}

/// Convert function decorated by #[kernel] into a single `lib.rs` for PTX-builder
pub fn func2kernel(func: &Function) -> String {
    let mut builder = func.create_builder();
    let lib = format!("{}\n{}", header(&builder), kernel(func));
    builder.compile(&lib).expect("Failed to compile")
}

pub fn func2caller(ptx_str: &str, func: &Function) -> TokenStream {
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
