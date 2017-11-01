#![feature(proc_macro)]

extern crate derive_kernel;

use derive_kernel::kernel;

#[kernel]
pub fn add(_a: &[f32], _b: &[f32], _c: &mut [f32]) {}

fn main() {
    //
}
