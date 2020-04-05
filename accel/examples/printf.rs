use accel::*;
use accel_derive::kernel;
use anyhow::Result;

#[kernel]
pub fn print() {
    let i = accel_core::index();
    accel_core::println!("Hello from {}", i);
}

fn main() -> Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();
    print(&ctx, 1, 4, &())?;
    Ok(())
}
