#![feature(proc_macro)]

extern crate derive_kernel;

use derive_kernel::kernel;

#[kernel(i)]
pub fn add(a: &[f32], b: &[f32], c: &mut [f32]) {
    *c[0] = a[0] + b[0];
}

fn main() {
    //
}
