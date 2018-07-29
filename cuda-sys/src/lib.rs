pub mod cublas;
pub mod cuda;
pub mod cudart;
pub mod vector_types;

#[test]
fn cuda_version() {
    let mut d_ver = 0;
    unsafe {
        cuda::cuDriverGetVersion(&mut d_ver as *mut i32);
    }
    println!("driver version = {}", d_ver);
}

mod cuda_tests;
mod cudart_tests;
