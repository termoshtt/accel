use accel_derive::kernel;

#[kernel]
#[dependencies("accel-core" = "0.3.0-alpha.1")]
unsafe fn dependencies() {}

fn main() {}
