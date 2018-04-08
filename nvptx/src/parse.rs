use proc_macro::TokenStream;
use syn::*;
use std::path::*;
use std::env;

use config::{Crate, Depends};
use build::Builder;

#[derive(Debug)]
pub struct Function {
    pub attrs: Vec<Attribute>,
    pub ident: Ident,
    pub vis: Visibility,
    pub block: Box<Block>,
    pub unsafety: Option<token::Unsafe>,
    pub inputs: punctuated::Punctuated<FnArg, token::Comma>,
    pub output: ReturnType,
    pub fn_token: token::Fn,
}

impl Function {
    pub fn parse(func: TokenStream) -> Self {
        let ItemFn {
            attrs,
            ident,
            vis,
            block,
            decl,
            unsafety,
            ..
        } = ::syn::parse(func.clone()).unwrap();
        let FnDecl {
            inputs,
            output,
            fn_token,
            ..
        } = { *decl };
        Function {
            attrs,
            ident,
            vis,
            block,
            unsafety,
            inputs,
            output,
            fn_token,
        }
    }

    pub fn input_values(&self) -> Vec<&Pat> {
        self.inputs
            .iter()
            .map(|arg| match arg {
                &FnArg::Captured(ref val) => &val.pat,
                _ => unreachable!(""),
            })
            .collect()
    }

    pub fn create_builder(&self) -> Builder {
        parse_builder_attrs(&self.attrs)
    }
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
    let mut depends = Depends::new();
    let mut build_path = None;
    for attr in attrs.iter() {
        let path = &attr.path;
        let path = quote!{#path}.to_string();
        let tts = &attr.tts;
        let tts = quote!{#tts}.to_string();
        let attr = tts.trim_matches(PENE);
        match path.as_str() {
            "depends" => depends.push(depends_to_crate(&attr)),
            "depends_path" => depends.push(depends_path_to_crate(&attr)),
            "build_path" => build_path = Some(as_build_path(&attr)),
            "build_path_home" => build_path = Some(as_build_path_home(&attr)),
            _ => unreachable!("Unsupported attribute: {:?}", path),
        }
    }
    match build_path {
        Some(path) => Builder::with_path(path, depends),
        None => Builder::new(depends),
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
