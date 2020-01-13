use accel_derive::kernel;

#[kernel]
unsafe fn dependencies_default() {
    let _i = accel_core::index(); // accel-core exists
}

fn main() {}
