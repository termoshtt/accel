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
    builder.copy_triplet().unwrap();
    builder.build().expect("xargo failed");
    builder.link().expect("Link failed");
    let ptx = builder.load_ptx().expect("Cannot load PTX");
    println!("{}", ptx);
}
