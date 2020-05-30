use accel::*;

#[kernel]
fn f1(a: &i32) {}

#[kernel]
fn f2(a: &i32, b: &mut i32) {}
