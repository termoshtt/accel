use std::env;

fn find_library_paths() -> Vec<String> {
    match env::var("CUDA_LIBRARY_PATH") {
        Ok(path) => {
            let split_char = if cfg!(target_os="windows") {
                ";"
            }
            else {
                ":"
            };

            path.split(split_char).map(|s| s.to_owned()).collect::<Vec<_>>()
        },
        Err(_) => vec![]
    }
}

fn main() {
    for p in find_library_paths() {
        println!("cargo:rustc-link-search=native={}", p);
    }
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rerun-if-changed=build.rs");
}
