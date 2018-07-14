extern "platform-intrinsic" {
    pub fn nvptx_block_dim_x() -> i32;
    pub fn nvptx_block_dim_y() -> i32;
    pub fn nvptx_block_dim_z() -> i32;
    pub fn nvptx_block_idx_x() -> i32;
    pub fn nvptx_block_idx_y() -> i32;
    pub fn nvptx_block_idx_z() -> i32;
    pub fn nvptx_grid_dim_x() -> i32;
    pub fn nvptx_grid_dim_y() -> i32;
    pub fn nvptx_grid_dim_z() -> i32;
    pub fn nvptx_syncthreads() -> ();
    pub fn nvptx_thread_idx_x() -> i32;
    pub fn nvptx_thread_idx_y() -> i32;
    pub fn nvptx_thread_idx_z() -> i32;
}

