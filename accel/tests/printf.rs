use accel::*;
use accel_derive::kernel;
use anyhow::Result;

#[kernel]
pub fn print() {
    let i = accel_core::index();
    accel_core::println!("Hello from {}", i);
}

#[test]
fn main() -> Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();
    let grid = Grid::x(1);
    let block = Block::x(4);
    print(&ctx, grid, block, &())?;
    Ok(())
}
