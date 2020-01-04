use failure::*;
use quote::ToTokens;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Default, Debug)]
pub struct Attributes {
    dependencies: HashMap<String, Depenency>,
}

pub fn parse_attrs(attrs: &[syn::Attribute]) -> Fallible<Attributes> {
    let mut kernel_attrs = Attributes::default();
    for attr in attrs {
        let path = attr.path.to_token_stream().to_string();
        match path.as_ref() {
            "dependencies" => {
                let dep = parse_dependency(
                    attr.tokens
                        .to_string()
                        .trim_start_matches('(')
                        .trim_end_matches(')'),
                )?;
                for (key, val) in dep {
                    kernel_attrs.dependencies.insert(key, val);
                }
            }
            _ => {
                bail!("Unsupported attribute: {}", path);
            }
        }
    }
    Ok(kernel_attrs)
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
