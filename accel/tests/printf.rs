use accel::*;
use accel_derive::kernel;

#[kernel]
pub fn print() {
    let i = accel_core::index();
    accel_core::println!("Hello from {}", i);
}

#[test]
fn main() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();
    print(ctx, 1, 4, &())?;
    Ok(())
}
