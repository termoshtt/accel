use accel::*;
use accel_derive::kernel;
use anyhow::Result;

#[kernel]
pub fn read_host_memory(ptr: *const i32) {
    accel_core::println!("Address = {:p}", ptr);
}

fn main() -> Result<()> {
    let device = driver::Device::nth(0)?;
    let ctx = device.create_context_auto()?;
    let grid = Grid::x(1);
    let block = Block::x(4);

    let i = 12345_i32;
    let p = &i as *const i32;
    println!("&i = {:p}", p);
    read_host_memory(&ctx, grid, block, &(&p,))?;
    Ok(())
}
