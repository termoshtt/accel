#![feature(proc_macro)]

extern crate accel;
extern crate accel_derive;

use accel_derive::kernel;
use accel::*;

#[kernel]
#[depends("accel-core" = "0.2.0-alpha")]
#[build_path_home(".rust2ptx")]
pub unsafe fn add(a: &[f64], b: &[f64], c: &mut [f64]) {
    let i = accel_core::index() as usize;
    let n = c.len();
    if i < n {
        c[i] = a[i] + b[i]
    }
}

fn main() {
    let n = 32;
    let mut a = UVec::new(n).unwrap();
    let mut b = UVec::new(n).unwrap();
    let mut c = UVec::new(n).unwrap();

    for i in 0..n {
        a[i] = i as f64;
        b[i] = 2.0 * i as f64;
    }
    println!("a = {:?}", a.as_slice());
    println!("b = {:?}", b.as_slice());

    let grid = Grid::x(1);
    let block = Block::x(n as u32);
    add(grid, block, &a, &b, &mut c);

    device::sync().unwrap();
    println!("c = {:?}", c.as_slice());
}
