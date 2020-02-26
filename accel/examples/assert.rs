use accel::*;
use accel_derive::kernel;

#[kernel]
pub fn assert() {
    accel_core::assert_eq!(1 + 2, 4);
}

fn main() -> anyhow::Result<()> {
    let grid = Grid::x(1);
    let block = Block::x(4);
    assert(grid, block)?;
    Ok(())
}
