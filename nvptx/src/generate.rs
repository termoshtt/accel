use syn::{Ident, ItemFn};
use parse::parse_builder_attrs;
use config::Crate;

pub fn header(crates: &[Crate]) -> String {
    let crates: Vec<Ident> = crates.iter().map(|c| Ident::from(c.name().replace("-", "_"))).collect();
    let tt = quote!{
        #![feature(abi_ptx)]
        #![no_std]
        #(extern crate #crates;), *
    };
    tt.to_string()
}

pub fn kernel(func: &ItemFn) -> String {
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
pub fn func2kernel(func: &ItemFn) -> String {
    let mut builder = parse_builder_attrs(&func.attrs);
    let lib = format!("{}\n{}", header(&builder.crates()), kernel(func));
    builder.compile(&lib)
}
