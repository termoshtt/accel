pub mod vector_types;
pub mod cuda;
pub mod cuda_runtime;
pub mod cublas;

#[test]
fn cuda_version() {
    let mut d_ver = 0;
    unsafe {
        cuda::cuDriverGetVersion(&mut d_ver as *mut i32);
    }
    println!("driver version = {}", d_ver);
}
