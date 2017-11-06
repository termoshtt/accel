#![feature(proc_macro)]

extern crate acc;
extern crate acc_derive;

use acc_derive::kernel;

#[kernel]
pub fn add(a: &[f32], b: &[f32], c: &mut [f32]) {
    *c[0] = a[0] + b[0];
}

fn main() {
    //
}
