use std::env;

fn main() {
    match env::var("CUDA_LIBRARY_PATH") {
        Ok(path) => for p in path.split(":") {
            println!("cargo:rustc-link-search=native={}", p);
        },
        Err(_) => {}
    };
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rerun-if-changed=build.rs");
}
