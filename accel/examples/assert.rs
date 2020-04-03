use accel::*;
use accel_derive::kernel;
use anyhow::Result;

#[kernel]
pub fn assert() {
    accel_core::assert_eq!(1 + 2, 4);
}

fn main() -> Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();
    let grid = Grid::x(1);
    let block = Block::x(4);
    assert(&ctx, grid, block, &())?;
    assert!(ctx.sync().is_err()); // This sync will fail because assertion in kernel code failed
    Ok(())
}
