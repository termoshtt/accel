//! Compiler driver for Rust to PTX assembler

use crate::parser::*;
use failure::*;
use quote::quote;
use std::process::Command;

const NIGHTLY_VERSION: &'static str = "nightly-2020-01-01";

/// Setup nightly rustc+cargo and nvptx64-nvidia-cuda target
fn rustup() -> Fallible<()> {
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

/// Generate Rust code for nvptx64-nvidia-cuda target from tokens
fn ptx_kernel(func: &syn::ItemFn) -> String {
    let vis = &func.vis;
    let ident = &func.sig.ident;
    let unsafety = &func.sig.unsafety;
    let block = &func.block;

    let fn_token = &func.sig.fn_token;
    let inputs = &func.sig.inputs;
    let output = &func.sig.output;

    let kernel = quote! {
        #![feature(abi_ptx)]
        #![no_std]
        #[no_mangle]
        #vis #unsafety extern "ptx-kernel" #fn_token #ident(#inputs) #output #block
    };
    kernel.to_string()
}

pub struct Driver {
    attrs: Attributes,
}

impl Driver {
    pub fn from_kernel(func: &syn::ItemFn) -> Fallible<Self> {
        rustup()?;
        let attrs = parse_attrs(&func.attrs)?;
        Ok(Driver { attrs })
    }

    fn compile(&self, _rust_str: &str) -> Fallible<String> {
        unimplemented!()
    }

    pub fn compile_tokens(&self, func: &syn::ItemFn) -> Fallible<String> {
        let lib_rs = ptx_kernel(func);
        self.compile(&lib_rs)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn rustup() {
        super::rustup().unwrap();
    }
}
