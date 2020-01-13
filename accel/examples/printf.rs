use accel::*;
use accel_derive::kernel;

#[kernel]
pub unsafe fn print() {
    core::arch::nvptx::vprintf("Hello GPU World!".as_ptr(), core::ptr::null_mut());
}

fn main() {
    let grid = Grid::x(1);
    let block = Block::x(4);
    print(grid, block).expect("print kernel failed");
}
