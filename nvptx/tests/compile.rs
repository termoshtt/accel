extern crate nvptx;

use nvptx::config::Crate;
use nvptx::compile::Builder;

const GPU_CODE: &'static str = r#"
#![feature(abi_ptx)]
#![no_std]
extern crate accel_core;
#[no_mangle]
pub unsafe extern "ptx-kernel" fn add(a: *const f64, b: *const f64, c: *mut f64, n: usize) {
    let i = accel_core::index();
    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}
"#;

#[test]
fn compile_tmp() {
    let crates = &[Crate::with_version("accel-core", "0.2.0-alpha")];
    let mut builder = Builder::new(crates);
    let ptx = builder.compile(GPU_CODE).unwrap();
    println!("PTX = {:?}", ptx);
}

#[test]
fn compile_path() {
    let crates = &[Crate::with_version("accel-core", "0.2.0-alpha")];
    let mut builder = Builder::with_path("~/tmp/rust2ptx", crates);
    let ptx = builder.compile(GPU_CODE).unwrap();
    println!("PTX = {:?}", ptx);
}
