use proc_macro2::TokenStream;
use quote::quote;
fn get_input_types(func: &syn::ItemFn) -> Vec<syn::Type> {
    func.sig
        .inputs
        .iter()
        .map(|arg| match arg {
            &syn::FnArg::Typed(ref val) => &*val.ty,
            _ => panic!("Unsupported kernel input type sigunature"),
        })
        .cloned()
        .collect()
}

fn get_input_arg_names(func: &syn::ItemFn) -> Vec<syn::Ident> {
    func.sig
        .inputs
        .iter()
        .map(|arg| match arg {
            &syn::FnArg::Typed(ref val) => match *val.pat {
                syn::Pat::Ident(ref pat) => &pat.ident,
                _ => panic!("Unsupported Input sigunature"),
            },
            _ => panic!("Unsupported Input sigunature"),
        })
        .cloned()
        .collect()
}

fn impl_submodule(ptx_str: &str, func: &syn::ItemFn) -> TokenStream {
    let ident = &func.sig.ident;
    let input_types = get_input_types(func);
    let kernel_name = quote! { #ident }.to_string();
    quote! {
        /// Auto-generated by accel-derive
        mod #ident {
            use ::accel::driver::{module, context};
            pub const PTX_STR: &'static str = #ptx_str;

            pub struct Module<'ctx>(module::Module<'ctx>);

            impl<'ctx> Module<'ctx> {
                pub fn new(ctx: &'ctx context::Context) -> ::anyhow::Result<Self> {
                    Ok(Module(module::Module::from_str(ctx, PTX_STR)?))
                }
            }

            impl<'ctx> module::Launchable for Module<'ctx> {
                type Args = (#(#input_types),*);
                fn get_kernel(&self) -> ::anyhow::Result<module::Kernel> {
                    Ok(self.0.get_kernel(#kernel_name)?)
                }
            }
        }
    }
}

fn caller(func: &syn::ItemFn) -> TokenStream {
    let vis = &func.vis;
    let ident = &func.sig.ident;
    let fn_token = &func.sig.fn_token;
    let inputs = &func.sig.inputs;
    let input_values = get_input_arg_names(func);
    let kernel_name = quote! { #ident }.to_string();
    quote! {
        #vis #fn_token #ident(
            ctx: & ::accel::driver::context::Context,
            grid: ::accel::Grid,
            block: ::accel::Block,
            #inputs
        ) -> ::anyhow::Result<()> {
            let module = ::accel::driver::module::Module::from_str(&ctx, #ident::PTX_STR)?;
            let mut kernel = module.get_kernel(#kernel_name)?;
            let mut args = [#(::accel::driver::module::void_cast(&#input_values)),*];
            unsafe { kernel.launch(args.as_mut_ptr(), grid, block)? };
            Ok(())
        }
    }
}

pub fn func2caller(ptx_str: &str, func: &syn::ItemFn) -> proc_macro::TokenStream {
    let impl_submodule = impl_submodule(ptx_str, func);
    let caller = caller(func);
    let code = quote! {
        #impl_submodule
        #caller
    };
    code.into()
}

#[cfg(test)]
mod tests {
    use crate::pretty_print;
    use anyhow::Result;

    const TEST_KERNEL: &'static str = r#"
    fn kernel_name(arg1: i32, arg2: f64) {}
    "#;

    #[test]
    fn arg_names() -> Result<()> {
        let func: syn::ItemFn = syn::parse_str(TEST_KERNEL)?;
        let args = super::get_input_arg_names(&func);
        assert_eq!(args[0].to_string(), "arg1");
        assert_eq!(args[1].to_string(), "arg2");
        Ok(())
    }

    #[test]
    fn impl_submodule() -> Result<()> {
        let func: syn::ItemFn = syn::parse_str(TEST_KERNEL)?;
        let ts = super::impl_submodule("", &func);
        pretty_print(&ts)?;
        Ok(())
    }

    #[test]
    fn caller() -> Result<()> {
        let func: syn::ItemFn = syn::parse_str(TEST_KERNEL)?;
        let ts = super::caller(&func);
        pretty_print(&ts)?;
        Ok(())
    }
}
