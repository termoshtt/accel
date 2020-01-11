use accel_derive::kernel;

#[kernel]
#[dependencies("accel-core" = "0.3.0-alpha.1")]
unsafe fn version() {
    let _i = accel_core::index();
}

#[kernel]
#[dependencies("accel-core" = { version = "0.3.0-alpha.1" })]
unsafe fn version_table() {
    let _i = accel_core::index();
}

fn main() {}
