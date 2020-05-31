use proc_macro2::*;
use quote::quote;
use syn::*;

fn seek_context_ident(input: &DeriveInput) -> Ident {
    match &input.data {
        syn::Data::Struct(syn::DataStruct { fields, .. }) => match fields {
            Fields::Named(fields_named) => {
                for field in fields_named.named.iter() {
                    let field = field.ident.clone().unwrap();
                    if field.to_string() == "context" || field.to_string() == "ctx" {
                        return field;
                    }
                }
            }
            _ => unreachable!("Must be named field"),
        },
        _ => unreachable!("Must be a struct"),
    };
    unreachable!("context or ctx not found")
}

pub fn contexted(input: DeriveInput) -> TokenStream {
    let name = &input.ident;
    let generics = &input.generics;
    let context_ident = seek_context_ident(&input);
    quote! {
        impl #generics Contexted for #name #generics {
            fn sync(&self) -> Result<()> {
                self.#context_ident.sync()
            }

            fn version(&self) -> Result<u32> {
                self.#context_ident.version()
            }

            fn guard(&self) -> Result<ContextGuard> {
                self.#context_ident.guard()
            }

            fn get_ref(&self) -> ContextRef {
                self.#context_ident.get_ref()
            }
        }
    }
}
