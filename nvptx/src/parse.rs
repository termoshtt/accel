use proc_macro::TokenStream;
use syn::*;
use std::path::*;
use config::Crate;
use compile::Builder;

pub fn parse_func(func: TokenStream) -> ItemFn {
    parse(func).expect("Not a function")
}

/// Parse attributes of kernel
///
/// For attributes are allowed:
///
/// - `crate`: add dependent crate
///    - `#[crate("accel-core")]` equals to `accel-core = "*"` in Cargo.toml
///    - `#[crate("accel-core" = "0.1.0")]` equals to `accel-core = "0.1.0"`
/// - `crate_path`: add dependent crate from local
///    - `#[crate_path("accel-core" = "/some/path")]`
///      equals to `accel-core = { path = "/some/path" }`
/// - `#[build_path("/some/path")]`: build PTX on "/some/path"
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
            "crate" => crates.push(as_crate(&attr)),
            "crate_path" => crates.push(as_crate_path(&attr)),
            "build_path" => build_path = Some(as_build_path(&attr)),
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

fn as_crate(dep: &str) -> Crate {
    let tokens: Vec<_> = dep.split('=').map(|s| s.trim_matches(QUOTE)).collect();
    match tokens.len() {
        // #[crate("accel-core")] case
        1 => Crate::new(tokens[0]),
        // #[crate("accel-core" = "0.1.0")] case
        2 => Crate::with_version(tokens[0], tokens[1]),
        _ => unreachable!("Invalid line: {}", dep),
    }
}

fn as_crate_path(dep: &str) -> Crate {
    let tokens: Vec<_> = dep.split('=').map(|s| s.trim_matches(QUOTE)).collect();
    match tokens.len() {
        // #[depends_path("accel-core" = "/some/path")] case
        2 => Crate::with_path(tokens[0], tokens[1]),
        _ => unreachable!("Invalid line: {}", dep),
    }
}
