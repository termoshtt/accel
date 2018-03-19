use std::collections::HashMap;
use std::path::*;
use toml;

#[derive(Debug, NewType)]
pub struct Depends(Vec<Crate>);

impl Depends {
    pub fn new() -> Self {
        Depends(Vec::new())
    }
}

impl ToString for Depends {
    fn to_string(&self) -> String {
        let dependencies = self.iter()
            .cloned()
            .map(|c| {
                let name = c.name;
                let version = c.version.unwrap_or("*".to_string());
                let path = c.path.map(|p| p.to_str().unwrap().to_owned());
                (name, CrateInfo { version, path })
            })
            .collect();
        let cargo = CargoTOML {
            package: Package::default(),
            profile: Profile::default(),
            dependencies,
        };
        toml::to_string(&cargo).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct Crate {
    name: String,
    version: Option<String>,
    path: Option<PathBuf>,
}

impl Crate {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn from_depends_str(dep: &str) -> Self {
        let pat: &[_] = &[' ', '"'];
        let tokens: Vec<_> = dep.split('=').map(|s| s.trim_matches(pat)).collect();
        match tokens.len() {
            // #[depends("accel-core")]
            1 => Self {
                name: tokens[0].to_owned(),
                version: None,
                path: None,
            },
            // #[depends("accel-core" = "0.1.0")]
            2 => Self {
                name: tokens[0].to_owned(),
                version: Some(tokens[1].to_owned()),
                path: None,
            },
            _ => unreachable!("Invalid line: {}", dep),
        }
    }

    pub fn from_depends_path_str(dep: &str) -> Self {
        let pat: &[_] = &[' ', '"'];
        let tokens: Vec<_> = dep.split('=').map(|s| s.trim_matches(pat)).collect();
        match tokens.len() {
            // #[depends_path("accel-core" = "/some/path")]
            2 => Self {
                name: tokens[0].to_owned(),
                version: None,
                path: Some(PathBuf::from(tokens[1])),
            },
            _ => unreachable!("Invalid line: {}", dep),
        }
    }
}

#[derive(Serialize)]
struct CargoTOML {
    package: Package,
    profile: Profile,
    dependencies: Dependencies,
}

#[derive(Serialize)]
struct Package {
    name: String,
    version: String,
}

impl Default for Package {
    fn default() -> Self {
        Package {
            name: "ptx-builder".to_string(),
            version: "0.1.0".to_string(),
        }
    }
}

#[derive(Serialize)]
struct Profile {
    dev: DevProfile,
}

impl Default for Profile {
    fn default() -> Self {
        Profile {
            dev: DevProfile::default(),
        }
    }
}

#[derive(Serialize)]
struct DevProfile {
    debug: bool,
}

impl Default for DevProfile {
    fn default() -> Self {
        DevProfile { debug: false }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct CrateInfo {
    pub path: Option<String>,
    pub version: String,
}

type Dependencies = HashMap<String, CrateInfo>;
