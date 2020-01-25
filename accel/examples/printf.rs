use accel::*;
use accel_derive::kernel;

#[kernel]
#[dependencies("accel-core" = { git = "https://gitlab.com/termoshtt/accel", branch = "assert_macros" })]
pub unsafe fn print() {
    let i = accel_core::index();
    accel_core::println!("Hello from {}", i);
}

fn main() -> anyhow::Result<()> {
    let grid = Grid::x(1);
    let block = Block::x(4);
    let device = driver::Device::nth(0)?;
    let _ctx = device.create_context_auto()?;
    print(grid, block)?;
    Ok(())
}
