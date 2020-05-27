use proc_macro2::{Span, TokenStream};
use quote::quote;

pub fn generate(item: TokenStream) -> TokenStream {
    let literal: syn::LitInt = syn::parse2(item).unwrap();
    let n: usize = literal.base10_parse().unwrap();
    (0..=n)
        .into_iter()
        .map(|i| {
            let name = syn::Ident::new(&format!("ArgRef{}", i), Span::call_site());
            if i > 0 {
                let device_sends: Vec<syn::Ident> = (1..=i)
                    .into_iter()
                    .map(|k| syn::Ident::new(&format!("D{}", k), Span::call_site()))
                    .collect();
                let args: Vec<syn::Ident> = (1..=i)
                    .into_iter()
                    .map(|k| syn::Ident::new(&format!("arg{}", k), Span::call_site()))
                    .collect();

                let generics_def = quote! { <'arg, #(#device_sends : DeviceSend),*> };
                let generics_use = quote! { <'arg, #(#device_sends),*> };
                let tuple = quote! { ( #(&'arg #device_sends,)* ) };

                quote! {
                    /// Auto-generated type by `accel_derive::define_argref` macro
                    #[repr(C)]
                    pub struct #name #generics_def {
                        #(pub #args : &'arg #device_sends),*
                    }

                    unsafe impl #generics_def Send for #name #generics_use {}
                    unsafe impl #generics_def Sync for #name #generics_use {}

                    impl #generics_def ArgRef for #name #generics_use {}

                    impl #generics_def From<#tuple> for #name #generics_use {
                        fn from((#(#args,)*): #tuple) -> Self {
                            Self { #(#args),* }
                        }
                    }
                }
            } else {
                quote! {
                    /// Auto-generated type by `accel_derive::define_argref` macro
                    #[repr(C)]
                    pub struct #name <'arg> {
                        phantom: ::std::marker::PhantomData<&'arg ()>
                    }

                    unsafe impl Send for #name <'_> {}
                    unsafe impl Sync for #name <'_> {}

                    impl ArgRef for #name <'_> {}

                    impl From<()> for #name <'_> {
                        fn from(_: ()) -> Self {
                            Self { phantom: Default::default() }
                        }
                    }
                }
            }
        })
        .collect()
}
