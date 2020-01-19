use accel::*;
use accel_derive::kernel;

#[kernel]
pub unsafe fn print() {
    core::arch::nvptx::vprintf("Hello GPU World!".as_ptr(), core::ptr::null_mut());
}

fn main() -> anyhow::Result<()> {
    let grid = Grid::x(1);
    let block = Block::x(4);
    let device = driver::Device::new(0)?;
    let _ctx = device.create_context_auto()?;
    print(grid, block)?;
    Ok(())
}
