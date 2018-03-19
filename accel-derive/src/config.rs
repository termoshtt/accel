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

    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            version: None,
            path: None,
        }
    }

    pub fn with_version(name: &str, version: &str) -> Self {
        Self {
            name: name.to_string(),
            version: Some(version.to_string()),
            path: None,
        }
    }

    pub fn with_path<P: AsRef<Path>>(name: &str, path: P) -> Self {
        Self {
            name: name.to_string(),
            version: None,
            path: Some(path.as_ref().to_owned()),
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
    release: ReleaseProfile,
}

impl Default for Profile {
    fn default() -> Self {
        Profile {
            release: ReleaseProfile::default(),
        }
    }
}

#[derive(Serialize)]
struct ReleaseProfile {
    debug: bool,
    panic: String,
}

impl Default for ReleaseProfile {
    fn default() -> Self {
        ReleaseProfile {
            debug: false,
            panic: "abort".to_string(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct CrateInfo {
    pub path: Option<String>,
    pub version: String,
}

type Dependencies = HashMap<String, CrateInfo>;
