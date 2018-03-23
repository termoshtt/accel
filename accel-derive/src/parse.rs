use proc_macro::TokenStream;
use syn::*;
use std::path::*;
use std::env;

use config::*;

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
}

pub struct KernelAttribute {
    pub depends: Depends,
    pub build_path: Option<PathBuf>,
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
pub fn parse_attrs(func: &Function) -> KernelAttribute {
    let mut attrs = KernelAttribute {
        depends: Depends::new(),
        build_path: None,
    };
    for attr in &func.attrs {
        let path = &attr.path;
        let path = &quote!{#path}.to_string();
        let tts = &attr.tts;
        let tts = &quote!{#tts}.to_string();
        let pene: &[_] = &['(', ')'];
        let dep = tts.trim_matches(pene);
        match path as &str {
            "depends" => attrs.depends.push(depends_to_crate(dep)),
            "depends_path" => attrs.depends.push(depends_path_to_crate(dep)),
            "build_path" => attrs.build_path = Some(PathBuf::from(path)),
            "build_path_home" => attrs.build_path = Some(env::home_dir().expect("No home dir").join(path).to_owned()),
            _ => unreachable!("Unsupported attribute: {:?}", path),
        }
    }
    attrs
}

fn depends_to_crate(dep: &str) -> Crate {
    let pat: &[_] = &[' ', '"'];
    let tokens: Vec<_> = dep.split('=').map(|s| s.trim_matches(pat)).collect();
    match tokens.len() {
        // #[depends("accel-core")] case
        1 => Crate::new(tokens[0]),
        // #[depends("accel-core" = "0.1.0")] case
        2 => Crate::with_version(tokens[0], tokens[1]),
        _ => unreachable!("Invalid line: {}", dep),
    }
}

fn depends_path_to_crate(dep: &str) -> Crate {
    let pat: &[_] = &[' ', '"'];
    let tokens: Vec<_> = dep.split('=').map(|s| s.trim_matches(pat)).collect();
    match tokens.len() {
        // #[depends_path("accel-core" = "/some/path")] case
        2 => Crate::with_path(tokens[0], tokens[1]),
        _ => unreachable!("Invalid line: {}", dep),
    }
}
