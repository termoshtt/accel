use failure::*;
use nvptx::manifest::Crate;
use proc_macro2::Span;
use quote::quote;

use crate::parser::Attributes;
use std::process::Command;

const NIGHTLY_VERSION: &'static str = "nightly-2020-01-01";

/// Setup nightly rustc+cargo and nvptx64-nvidia-cuda target
pub fn rustup() -> Fallible<()> {
    let st = Command::new("rustup")
        .args(&[
            "toolchain",
            "install",
            NIGHTLY_VERSION,
            "--profile",
            "minimal",
        ])
        .status()?;
    if !st.success() {
        bail!("Cannot get nightly toolchain by rustup: {:?}", st);
    }

    let st = Command::new("rustup")
        .args(&[
            "target",
            "add",
            "nvptx64-nvidia-cuda",
            "--toolchain",
            NIGHTLY_VERSION,
        ])
        .status()?;
    if !st.success() {
        bail!("Cannot get nvptx64-nvidia-cuda target: {:?}", st);
    }
    Ok(())
}

/// Header part of lib.rs
fn header(crates: &[Crate]) -> String {
    let crates: Vec<syn::Ident> = crates
        .iter()
        .map(|c| syn::Ident::new(&c.name.replace("-", "_"), Span::call_site()))
        .collect();
    let tt = quote! {
        #![feature(abi_ptx)]
        #![no_std]
        #(extern crate #crates;), *
    };
    tt.to_string()
}

/// Kernel part of lib.rs
fn ptx_kernel(func: &syn::ItemFn) -> String {
    let vis = &func.vis;
    let ident = &func.sig.ident;
    let unsafety = &func.sig.unsafety;
    let block = &func.block;

    let fn_token = &func.sig.fn_token;
    let inputs = &func.sig.inputs;
    let output = &func.sig.output;

    let kernel = quote! {
        #[no_mangle]
        #vis #unsafety extern "ptx-kernel" #fn_token #ident(#inputs) #output #block
    };
    kernel.to_string()
}

/// Convert #[kernel] function into lib.rs
pub fn func2kernel(func: &syn::ItemFn) -> String {
    let attrs = Attributes::parse(&func.attrs);
    let driver = attrs.create_driver();
    let lib_rs = format!("{}\n{}", header(attrs.get_crates()), ptx_kernel(func));
    driver.compile_str(&lib_rs).expect("Failed to compile")
}

#[cfg(test)]
mod tests {
    #[test]
    fn rustup() {
        super::rustup().unwrap();
    }
}
