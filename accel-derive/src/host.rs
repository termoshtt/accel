use inflections::Inflect;
use proc_macro::TokenStream;
use proc_macro2::Span;
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
    let ptx_const_name = syn::Ident::new(
        &format!("{}_ptx", kernel_name).to_constant_case(),
        Span::call_site(),
    );

    let caller = quote! {
        #vis const #ptx_const_name: &'static str = #ptx_str;
        #vis #fn_token #ident(grid: accel::Grid, block: accel::Block, #inputs) -> anyhow::Result<()> {
            use accel::driver::{
                kernel::void_cast,
                module::Module,
                device::Device,
            };
            let device = Device::nth(0)?;
            let ctx = device.create_context_auto()?;
            let module = Module::from_str(&ctx, #ptx_const_name)?;
            let mut kernel = module.get_kernel(#kernel_name)?;
            let mut args = [#(void_cast(&#input_values)),*];
            unsafe { kernel.launch(args.as_mut_ptr(), grid, block)? };
            Ok(())
        }
    };
    caller.into()
}
