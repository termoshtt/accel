use accel::*;
use accel_derive::kernel;

#[kernel]
#[dependencies("accel-core" = { git = "https://gitlab.com/termoshtt/accel", branch = "assert_macros" })]
pub fn assert() {
    accel_core::assert_eq!(1 + 2, 4);
}

fn main() -> anyhow::Result<()> {
    let grid = Grid::x(1);
    let block = Block::x(4);
    let device = driver::Device::nth(0)?;
    let _ctx = device.create_context_auto()?;
    assert(grid, block)?;
    Ok(())
}
