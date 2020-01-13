use failure::*;
use maplit::hashmap;
use quote::ToTokens;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize)]
pub struct MetaData {
    package: HashMap<&'static str, String>,
    lib: HashMap<&'static str, Vec<&'static str>>,
    dependencies: HashMap<String, Depenency>,
}

impl MetaData {
    fn new(name: &str) -> Self {
        MetaData {
            package: hashmap! { "version" => "0.0.0".into(), "name" => name.into(), "edition" => "2018".into() },
            lib: hashmap! { "crate-type" => vec![ "cdylib" ] },
            dependencies: HashMap::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.package["name"]
    }

    pub fn from_token(func: &syn::ItemFn) -> Fallible<Self> {
        let attrs = &func.attrs;
        let mut kernel_attrs = MetaData::new(&func.sig.ident.to_string());
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
                "name" => {
                    let token = attr.tokens.to_string();
                    let name = token.trim_start_matches('(').trim_end_matches(')').trim();
                    kernel_attrs.package.insert("name", name.into());
                }
                _ => {
                    bail!("Unsupported attribute: {}", path);
                }
            }
        }
        kernel_attrs
            .dependencies
            .entry("accel-core".into())
            .or_insert(Depenency::Version("0.3.0-alpha.1".into()));
        Ok(kernel_attrs)
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
    Ok(toml::from_str(&dep.replace("\n", ""))?)
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

        let map = super::parse_dependency(
            r#"accel-core = { git = "https://github.com/rust-accel/accel", branch = "master" }"#,
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
