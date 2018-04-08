use proc_macro::TokenStream;
use syn::*;
use std::path::*;
use std::env;

use config::Crate;
use compile::Builder;

pub fn parse_func(func: TokenStream) -> ItemFn {
    parse(func).expect("Not a function")
}

/// Parse attributes of kernel
///
/// For attributes are allowed:
///
/// - `depends`: add dependent crate
///    - `#[depends("accel-core")]` equals to `accel-core = "*"` in Cargo.toml
///    - `#[depends("accel-core" = "0.1.0")]` equals to `accel-core = "0.1.0"`
/// - `depends_path`: add dependent crate from local
///    - `#[depends_path("accel-core" = "/some/path")]`
///      equals to `accel-core = { path = "/some/path" }`
/// - `#[build_path("/some/path")]`: build PTX on "/some/path"
/// - `#[build_path_home("path/to/work")]`: build PTX on "$HOME/path/to/work"
///
pub fn parse_builder_attrs(attrs: &[Attribute]) -> Builder {
    let mut crates = Vec::new();
    let mut build_path = None;
    for attr in attrs.iter() {
        let path = &attr.path;
        let path = quote!{#path}.to_string();
        let tts = &attr.tts;
        let tts = quote!{#tts}.to_string();
        let attr = tts.trim_matches(PENE);
        match path.as_str() {
            "crates" => crates.push(depends_to_crate(&attr)),
            "depends_path" => crates.push(depends_path_to_crate(&attr)),
            "build_path" => build_path = Some(as_build_path(&attr)),
            "build_path_home" => build_path = Some(as_build_path_home(&attr)),
            _ => unreachable!("Unsupported attribute: {:?}", path),
        }
    }
    match build_path {
        Some(path) => Builder::with_path(path, &crates),
        None => Builder::new(&crates),
    }
}

const PENE: &[char] = &['(', ')'];
const QUOTE: &[char] = &[' ', '"'];

fn as_build_path(path: &str) -> PathBuf {
    PathBuf::from(path.trim_matches(QUOTE))
}

fn as_build_path_home(path: &str) -> PathBuf {
    let path = path.trim_matches(QUOTE);
    env::home_dir().expect("No home dir").join(path).to_owned()
}

fn depends_to_crate(dep: &str) -> Crate {
    let tokens: Vec<_> = dep.split('=').map(|s| s.trim_matches(QUOTE)).collect();
    match tokens.len() {
        // #[depends("accel-core")] case
        1 => Crate::new(tokens[0]),
        // #[depends("accel-core" = "0.1.0")] case
        2 => Crate::with_version(tokens[0], tokens[1]),
        _ => unreachable!("Invalid line: {}", dep),
    }
}

fn depends_path_to_crate(dep: &str) -> Crate {
    let tokens: Vec<_> = dep.split('=').map(|s| s.trim_matches(QUOTE)).collect();
    match tokens.len() {
        // #[depends_path("accel-core" = "/some/path")] case
        2 => Crate::with_path(tokens[0], tokens[1]),
        _ => unreachable!("Invalid line: {}", dep),
    }
}
