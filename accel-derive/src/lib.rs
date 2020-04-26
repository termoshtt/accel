#![recursion_limit = "128"]

//! Get compiled PTX as `String`
//! ----------------------------
//!
//! The proc-macro `#[kernel]` creates a submodule `add::` in addition to a function `add`.
//! Kernel Rust code is compiled into PTX string using rustc's `nvptx64-nvidia-cuda` toolchain.
//! Generated PTX string is embedded into proc-macro output as `{kernel_name}::PTX_STR`.
//!
//! ```
//! use accel_derive::kernel;
//!
//! #[kernel]
//! unsafe fn add(a: *const f64, b: *const f64, c: *mut f64, n: usize) {
//!     let i = accel_core::index();
//!     if (i as usize) < n {
//!         *c.offset(i) = *a.offset(i) + *b.offset(i);
//!     }
//! }
//!
//! // PTX assembler code is embedded as `add::PTX_STR`
//! println!("{}", add::PTX_STR);
//! ```

mod builder;
mod host;
mod parser;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, func: TokenStream) -> TokenStream {
    let func: syn::ItemFn = syn::parse(func).expect("Not a function");
    let ptx_str = builder::compile_tokens(&func).expect("Failed to compile to PTX");
    host::func2caller(&ptx_str, &func).into()
}
