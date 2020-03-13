#![recursion_limit = "128"]

extern crate proc_macro;

mod builder;
mod host;
mod parser;

use anyhow::Result;
use proc_macro::TokenStream;
use std::{
    io::Write,
    process::{Command, Stdio},
};

use host::*;

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, func: TokenStream) -> TokenStream {
    let func: syn::ItemFn = syn::parse(func).expect("Not a function");
    let ptx_str = builder::compile_tokens(&func).expect("Failed to compile to PTX");
    func2caller(&ptx_str, &func)
}

/// Format TokenStream by rustfmt
///
/// This can test if the input TokenStream is valid in terms of rustfmt.
#[allow(dead_code)]
pub(crate) fn pretty_print(tt: &impl ToString) -> Result<()> {
    let mut fmt = Command::new("rustfmt")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    fmt.stdin
        .as_mut()
        .unwrap()
        .write(tt.to_string().as_bytes())?;
    let out = fmt.wait_with_output()?;
    println!("{}", String::from_utf8_lossy(&out.stdout));
    Ok(())
}
