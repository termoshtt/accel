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

    let result = assert(&ctx, grid, block, &());
    assert!(result.is_err()); // assertion failed
    Ok(())
}
