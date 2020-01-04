use failure::*;
use nvptx::manifest::Crate;
use quote::quote;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

pub struct Attributes {
    crates: Vec<Crate>,
}

impl Attributes {
    pub fn parse(attrs: &[syn::Attribute]) -> Self {
        Self {
            crates: attrs.iter().map(|attr| parse_crate(attr)).collect(),
        }
    }

    pub fn get_crates(&self) -> &[Crate] {
        &self.crates
    }

    /// Create a nvptx compiler-driver
    pub fn create_driver(&self) -> nvptx::Driver {
        let driver = nvptx::Driver::new().expect("Fail to create compiler-driver");
        nvptx::manifest::generate(driver.path(), &self.crates)
            .expect("Fail to generate Cargo.toml");
        driver
    }
}

const PENE: &[char] = &['(', ')'];
const QUOTE: &[char] = &[' ', '"'];

/// Parse attributes of kernel
///
/// - `crate`: add dependent crate
///    - `#[crate("accel-core")]` equals to `accel-core = "*"` in Cargo.toml
///    - `#[crate("accel-core" = "0.1.0")]` equals to `accel-core = "0.1.0"`
/// - `crate_path`: add dependent crate from local
///    - `#[crate_path("accel-core" = "/some/path")]`
///      equals to `accel-core = { path = "/some/path" }`
pub fn parse_crate(attr: &syn::Attribute) -> Crate {
    let path = &attr.path;
    let path = quote! {#path}.to_string();
    let tts = attr.tokens.to_string();
    let tokens: Vec<_> = tts
        .trim_matches(PENE)
        .split('=')
        .map(|s| s.trim_matches(QUOTE).to_string())
        .collect();
    match path.as_str() {
        "crate" => {
            match tokens.len() {
                // #[crate("accel-core")] case
                1 => Crate {
                    name: tokens[0].clone(),
                    version: None,
                    path: None,
                },
                // #[crate("accel-core" = "0.1.0")] case
                2 => Crate {
                    name: tokens[0].clone(),
                    version: Some(tokens[1].clone()),
                    path: None,
                },
                _ => unreachable!("Invalid line: {:?}", attr),
            }
        }
        "crate_path" => {
            match tokens.len() {
                // #[crate_path("accel-core" = "/some/path")] case
                2 => Crate {
                    name: tokens[0].clone(),
                    version: None,
                    path: Some(fs::canonicalize(&tokens[1]).expect("Fail to normalize")),
                },
                _ => unreachable!("Invalid line: {:?}", attr),
            }
        }
        _ => unreachable!("Unsupported attribute: {:?}", path),
    }
}

// Should I use `cargo::core::dependency::Depenency`?
// https://docs.rs/cargo/0.41.0/cargo/core/dependency/struct.Dependency.html
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
enum Depenency {
    Version(String),
    VersionTable {
        version: String,
        #[serde(default)]
        features: Vec<String>,
    },
    Git {
        git: String,
        branch: Option<String>,
        tag: Option<String>,
        hash: Option<String>,
        #[serde(default)]
        features: Vec<String>,
    },
    Path {
        path: String,
        #[serde(default)]
        features: Vec<String>,
    },
}

fn parse_dependency(dep: &str) -> Fallible<HashMap<String, Depenency>> {
    Ok(toml::from_str(dep)?)
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse_dependency() {
        let map = super::parse_dependency(r#"accel-core = "0.1.1""#).unwrap();
        dbg!(map);
        let map = super::parse_dependency(r#"accel-core = { version = "0.1.1" }"#).unwrap();
        dbg!(map);

        let map = super::parse_dependency(
            r#"accel-core = { git = "https://github.com/rust-accel/accel" }"#,
        )
        .unwrap();
        dbg!(map);

        // `git` is lacked
        assert!(super::parse_dependency(r#"accel-core = { branch = "master" }"#,).is_err());

        // Unsupported tag
        assert!(super::parse_dependency(
            r#"accel-core = { git = "https://github.com/rust-accel/accel", homhom = "master" }"#,
        )
        .is_err());
    }
}
