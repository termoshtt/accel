use super::*;

pub fn index() -> isize {
    let block_id = block_idx().into_id(grid_dim());
    let thread_id = thread_idx().into_id(block_dim());
    (block_id + thread_id) as isize
}
