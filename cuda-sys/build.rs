
use std::env;

fn main() {
    match env::var("CUDA_LIBRARY_PATH") {
        Ok(path) => println!("cargo:rustc-link-search=native={}", path),
        Err(_) => {}
    };
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rerun-if-changed=build.rs");
}
