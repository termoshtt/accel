use accel_derive::kernel;

#[kernel]
#[dependencies("accel-core" = "0.3.0-alpha.1")]
unsafe fn version() {}

#[kernel]
#[dependencies("accel-core" = { version = "0.3.0-alpha.1" })]
unsafe fn version_table() {}

fn main() {}
