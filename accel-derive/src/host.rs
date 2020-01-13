use proc_macro::TokenStream;
use quote::quote;

pub fn func2caller(ptx_str: &str, func: &syn::ItemFn) -> TokenStream {
    let vis = &func.vis;
    let ident = &func.sig.ident;

    let fn_token = &func.sig.fn_token;
    let inputs = &func.sig.inputs;
    // FIXME should branch by output exists
    let _output = &func.sig.output;

    let input_values: Vec<_> = inputs
        .iter()
        .map(|arg| match arg {
            &syn::FnArg::Typed(ref val) => &val.pat,
            _ => unreachable!(""),
        })
        .collect();
    let kernel_name = quote! { #ident }.to_string();

    let caller = quote! {
        #vis #fn_token #ident(grid: accel::Grid, block: accel::Block, #inputs) -> Result<(), accel::error::AccelError> {
            use accel::{
                kernel::void_cast,
                driver::module::Module
            };
            let module = Module::from_str(#ptx_str)?;
            let mut kernel = module.get_kernel(#kernel_name)?;
            let mut args = [#(void_cast(&#input_values)),*];
            unsafe { kernel.launch(args.as_mut_ptr(), grid, block)? };
            Ok(())
        }
    };
    caller.into()
}
