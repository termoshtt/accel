extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn find_library_paths() -> Vec<String> {
    match env::var("CUDA_LIBRARY_PATH") {
        Ok(path) => {
            let split_char = if cfg!(target_os = "windows") { ";" } else { ":" };

            path.split(split_char).map(|s| s.to_owned()).collect::<Vec<_>>()
        }
        Err(_) => vec![],
    }
}

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindgen::builder()
        .header("wrapper/cuda.h")
        .whitelist_recursively(false)
        .whitelist_type("^CU.*")
        .whitelist_type("^cuuint(32|64)_t")
        .whitelist_type("^cudaError_enum")
        .whitelist_type("^cudaMem.*")
        .whitelist_var("^CU.*")
        .whitelist_function("^CU.*")
        .whitelist_function("^cu.*")
        .default_enum_style(bindgen::EnumVariation::Rust)
        .generate()
        .expect("Unable to generate CUDA bindings")
        .write_to_file(out_path.join("cuda_bindings.rs"))
        .expect("Unable to write CUDA bindings");

    bindgen::builder()
        .header("wrapper/cublas.h")
        .whitelist_recursively(false)
        .whitelist_type("^cublas.*")
        .whitelist_var("^cublas.*")
        .whitelist_function("^cublas.*")
        .default_enum_style(bindgen::EnumVariation::Rust)
        .generate()
        .expect("Unable to generate CUBLAS bindings")
        .write_to_file(out_path.join("cublas_bindings.rs"))
        .expect("Unable to write CUBLAS bindings");

    bindgen::builder()
        .header("wrapper/cucomplex.h")
        .whitelist_recursively(false)
        .whitelist_type("^cu.*Complex$")
        .default_enum_style(bindgen::EnumVariation::Rust)
        .generate()
        .expect("Unable to generate CUComplex bindings")
        .write_to_file(out_path.join("cucomplex_bindings.rs"))
        .expect("Unable to write CUComplex bindings");

    bindgen::builder()
        .header("wrapper/cudart.h")
        .whitelist_recursively(false)
        .whitelist_type("^cuda.*")
        .whitelist_type("^surfaceReference")
        .whitelist_type("^textureReference")
        .whitelist_var("^cuda.*")
        .whitelist_function("^cuda.*")
        .default_enum_style(bindgen::EnumVariation::Rust)
        .generate()
        .expect("Unable to generate CUDA RT bindings")
        .write_to_file(out_path.join("cudart_bindings.rs"))
        .expect("Unable to write CUDA RT bindings");

    bindgen::builder()
        .header("wrapper/driver_types.h")
        .whitelist_recursively(false)
        .whitelist_type("^CU.*")
        .whitelist_type("^cuda.*")
        .default_enum_style(bindgen::EnumVariation::Rust)
        .generate()
        .expect("Unable to generate driver types bindings")
        .write_to_file(out_path.join("driver_types_bindings.rs"))
        .expect("Unable to write driver types bindings");

    bindgen::builder()
        .header("wrapper/library_types.h")
        .whitelist_recursively(false)
        .whitelist_type("^cuda.*")
        .whitelist_type("^libraryPropertyType.*")
        .default_enum_style(bindgen::EnumVariation::Rust)
        .generate()
        .expect("Unable to generate library types bindings")
        .write_to_file(out_path.join("library_types_bindings.rs"))
        .expect("Unable to write library types bindings");

    bindgen::builder()
        .header("wrapper/vector_types.h")
        // .whitelist_recursively(false)
        .whitelist_type("^u?char[0-9]$")
        .whitelist_type("^dim[0-9]$")
        .whitelist_type("^double[0-9]$")
        .whitelist_type("^float[0-9]$")
        .whitelist_type("^u?int[0-9]$")
        .whitelist_type("^u?long[0-9]$")
        .whitelist_type("^u?longlong[0-9]$")
        .whitelist_type("^u?short[0-9]$")
        .default_enum_style(bindgen::EnumVariation::Rust)
        .derive_copy(true)
        .generate()
        .expect("Unable to generate vector types bindings")
        .write_to_file(out_path.join("vector_types_bindings.rs"))
        .expect("Unable to write vector types bindings");

    for p in find_library_paths() {
        println!("cargo:rustc-link-search=native={}", p);
    }
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rerun-if-changed=build.rs");
}
