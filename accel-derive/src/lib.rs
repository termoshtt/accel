/*!

Implementation of #[kernel] proc-macro
---------------------------------------

`#[kernel]` function will be converted to two part:

- Device code will be compiled into PTX assembler
- Host code which call the generated device code (PTX asm) using `accel::module` API

```ignore
use accel_derive::kernel;

#[kernel]
#[crate("accel-core" = "0.2.0-alpha")]
pub unsafe fn add(a: *const f64, b: *const f64, c: *mut f64, n: usize) {
    let i = accel_core::index();
    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}
```

will generate a crate with a device code

```ignore
#[no_mangle]
pub unsafe extern "ptx-kernel" fn add(a: *const f64, b: *const f64, c: *mut f64, n: usize) {
    let i = accel_core::index();
    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}
```

and `Cargo.toml`

```toml
[dependencies]
accel-core = "0.2.0-alpha"
```

This crate will be compiled into PTX assembler using `nvptx64-nvidia-cuda` target.

On the other hand, corresponding host code will also generated:

```ignore
use ::accel::{Grid, Block, kernel::void_cast, module::Module};

pub fn add(grid: Grid, block: Block, a: *const f64, b: *const f64, c: *mut f64, n: usize) {
    thread_local!{
        static module: Module = Module::from_str("{{PTX_STRING}}").expect("Load module failed");
    }
    module.with(|m| {
        let mut kernel = m.get_kernel("add").expect("Failed to get Kernel");
        let mut args = [void_cast(&a)), void_cast(&b), void_cast(&c)];
        unsafe {
            kernel.launch(args.as_mut_ptr(), grid, block).expect("Failed to launch kernel")
        };
    })
}
```

where `{{PTX_STRING}}` is a PTX assembler string compiled from the device code.
This can be called like following:

```ignore
let grid = Grid::x(1);
let block = Block::x(n as u32);
add(grid, block, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n);
```

*/

#![recursion_limit = "128"]

extern crate proc_macro;

use nvptx::manifest::Crate;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use std::fs;

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, func: TokenStream) -> TokenStream {
    let func = syn::parse(func).expect("Not a function");
    let ptx_str = func2kernel(&func);
    func2caller(&ptx_str, &func)
}

struct Attributes {
    crates: Vec<Crate>,
}

impl Attributes {
    fn parse(attrs: &[syn::Attribute]) -> Self {
        Self {
            crates: attrs.iter().map(|attr| parse_crate(attr)).collect(),
        }
    }

    fn get_crates(&self) -> &[Crate] {
        &self.crates
    }

    /// Create a nvptx compiler-driver
    fn create_driver(&self) -> nvptx::Driver {
        let driver = nvptx::Driver::new().expect("Fail to create compiler-driver");
        nvptx::manifest::generate(driver.path(), &self.crates)
            .expect("Fail to generate Cargo.toml");
        driver
    }
}

const PENE: &[char] = &['(', ')'];
const QUOTE: &[char] = &[' ', '"'];

/// Parse attributes of kernel
///
/// - `crate`: add dependent crate
///    - `#[crate("accel-core")]` equals to `accel-core = "*"` in Cargo.toml
///    - `#[crate("accel-core" = "0.1.0")]` equals to `accel-core = "0.1.0"`
/// - `crate_path`: add dependent crate from local
///    - `#[crate_path("accel-core" = "/some/path")]`
///      equals to `accel-core = { path = "/some/path" }`
fn parse_crate(attr: &syn::Attribute) -> Crate {
    let path = &attr.path;
    let path = quote! {#path}.to_string();
    let tts = attr.tokens.to_string();
    let tokens: Vec<_> = tts
        .trim_matches(PENE)
        .split('=')
        .map(|s| s.trim_matches(QUOTE).to_string())
        .collect();
    match path.as_str() {
        "crate" => {
            match tokens.len() {
                // #[crate("accel-core")] case
                1 => Crate {
                    name: tokens[0].clone(),
                    version: None,
                    path: None,
                },
                // #[crate("accel-core" = "0.1.0")] case
                2 => Crate {
                    name: tokens[0].clone(),
                    version: Some(tokens[1].clone()),
                    path: None,
                },
                _ => unreachable!("Invalid line: {:?}", attr),
            }
        }
        "crate_path" => {
            match tokens.len() {
                // #[crate_path("accel-core" = "/some/path")] case
                2 => Crate {
                    name: tokens[0].clone(),
                    version: None,
                    path: Some(fs::canonicalize(&tokens[1]).expect("Fail to normalize")),
                },
                _ => unreachable!("Invalid line: {:?}", attr),
            }
        }
        _ => unreachable!("Unsupported attribute: {:?}", path),
    }
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
fn func2kernel(func: &syn::ItemFn) -> String {
    let attrs = Attributes::parse(&func.attrs);
    let driver = attrs.create_driver();
    let lib_rs = format!("{}\n{}", header(attrs.get_crates()), ptx_kernel(func));
    driver.compile_str(&lib_rs).expect("Failed to compile")
}

fn func2caller(ptx_str: &str, func: &syn::ItemFn) -> TokenStream {
    let vis = &func.vis;
    let ident = &func.sig.ident;

    let fn_token = &func.sig.fn_token;
    let inputs = &func.sig.inputs;
    let output = &func.sig.output;

    let input_values: Vec<_> = inputs
        .iter()
        .map(|arg| match arg {
            &syn::FnArg::Typed(ref val) => &val.pat,
            _ => unreachable!(""),
        })
        .collect();
    let kernel_name = quote! { #ident }.to_string();

    let caller = quote! {
        #vis #fn_token #ident(grid: ::accel::Grid, block: ::accel::Block, #inputs) #output {
            use ::accel::kernel::void_cast;
            use ::accel::module::Module;
            thread_local!{
                static module: Module = Module::from_str(#ptx_str).expect("Load module failed");
            }
            module.with(|m| {
                let mut kernel = m.get_kernel(#kernel_name).expect("Failed to get Kernel");
                let mut args = [#(void_cast(&#input_values)),*];
                unsafe { kernel.launch(args.as_mut_ptr(), grid, block).expect("Failed to launch kernel") };
            })
        }
    };
    caller.into()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        // unimplemented!()
    }
}
