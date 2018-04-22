extern crate nvptx;

use nvptx::compile::Builder;
use std::env;
use std::path::*;

fn get_manifest_path() -> PathBuf {
    let mut dir = env::current_dir().unwrap();
    loop {
        let manif = dir.join("Cargo.toml");
        if manif.exists() {
            return dir;
        }
        dir = match dir.parent() {
            Some(dir) => dir.to_owned(),
            None => panic!("Cargo.toml cannot found"),
        };
    }
}

fn main() {
    let manifest_path = get_manifest_path();
    let builder = Builder::exists(manifest_path);
    builder.copy_triplet();
    builder.build();
    builder.link();
    let ptx = builder.load_ptx();
    println!("{}", ptx);
}
