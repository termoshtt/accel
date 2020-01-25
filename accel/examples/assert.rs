use accel::*;
use accel_derive::kernel;

#[kernel]
pub unsafe fn assert() {
    let msg = "Assertion";
    let filename = file!();
    let line = line!();
    let func_name = ""; // cannot get function name. See https://github.com/rust-lang/rfcs/pull/2818
    core::arch::nvptx::__assert_fail(msg.as_ptr(), filename.as_ptr(), 0, func_name.as_ptr());
}

fn main() -> anyhow::Result<()> {
    let grid = Grid::x(1);
    let block = Block::x(4);
    let device = driver::Device::nth(0)?;
    let _ctx = device.create_context_auto()?;
    assert(grid, block)?;
    Ok(())
}
