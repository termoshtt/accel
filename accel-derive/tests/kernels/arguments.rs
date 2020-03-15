//! Testing launch arguments are correctly handled

use accel::*;
use accel_derive::kernel;
use anyhow::Result;

#[kernel]
pub fn launch(i: i32) {
    accel_core::println!("i = {}", i);
}

fn test() -> Result<()> {
    let device = driver::Device::nth(0)?;
    let ctx = device.create_context_auto()?;
    let i = 12;
    let grid = Grid::x(1);
    let block = Block::x(4);
    launch(&ctx, grid, block, &(&i,))?;
    Ok(())
}

// Only check `test` can be compiled. not run here
fn main() {}
