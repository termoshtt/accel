//! proc_macro for accel's #[kernel]
#![feature(proc_macro)]
#![recursion_limit = "128"]

extern crate nvptx;
extern crate proc_macro;
#[macro_use]
extern crate quote;
extern crate syn;

use nvptx::manifest::Crate;
use proc_macro::TokenStream;
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
        nvptx::manifest::generate(driver.path(), &self.crates).expect("Fail to generate Cargo.toml");
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
    let path = quote!{#path}.to_string();
    let tts = &attr.tts;
    let tts = quote!{#tts}.to_string();
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
        .map(|c| syn::Ident::from(c.name.replace("-", "_")))
        .collect();
    let tt = quote!{
        #![feature(abi_ptx)]
        #![no_std]
        #(extern crate #crates;), *
    };
    tt.to_string()
}

/// Kernel part of lib.rs
fn ptx_kernel(func: &syn::ItemFn) -> String {
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

/// Convert #[kernel] function into lib.rs
fn func2kernel(func: &syn::ItemFn) -> String {
    let attrs = Attributes::parse(&func.attrs);
    let driver = attrs.create_driver();
    let lib_rs = format!("{}\n{}", header(attrs.get_crates()), ptx_kernel(func));
    driver.compile_str(&lib_rs).expect("Failed to compile")
}

fn func2caller(ptx_str: &str, func: &syn::ItemFn) -> TokenStream {
    let vis = &func.vis;
    let ident = &func.ident;

    let decl = &func.decl;
    let inputs = &decl.inputs;
    let output = &decl.output;
    let fn_token = &decl.fn_token;

    let input_values: Vec<_> = inputs
        .iter()
        .map(|arg| match arg {
            &syn::FnArg::Captured(ref val) => &val.pat,
            _ => unreachable!(""),
        })
        .collect();
    let kernel_name = quote!{ #ident }.to_string();

    let caller = quote!{
        mod ptx_mod {
            use ::std::cell::RefCell;
            use ::accel::module::Module;
            thread_local! {
                #[allow(non_upper_case_globals)]
                pub static #ident: RefCell<Module>
                    = RefCell::new(Module::from_str(#ptx_str).expect("Load module failed"));
            }
        }
        #vis #fn_token #ident(grid: ::accel::Grid, block: ::accel::Block, #inputs) #output {
            use ::accel::kernel::void_cast;
            ptx_mod::#ident.with(|m| {
                let m = m.borrow();
                let mut kernel = m.get_kernel(#kernel_name).expect("Failed to get Kernel");
                let mut args = [#(void_cast(&#input_values)),*];
                unsafe { kernel.launch(args.as_mut_ptr(), grid, block).expect("Failed to launch kernel") };
            })
        }
    };
    caller.into()
}
