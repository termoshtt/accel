
use std::collections::HashMap;
use toml;

#[derive(Serialize)]
pub struct CargoTOML {
    package: Package,
    profile: Profile,
    dependencies: Dependencies,
}

#[derive(Serialize)]
pub struct Package {
    name: String,
    version: String,
}

#[derive(Serialize)]
pub struct Profile {
    dev: DevProfile,
}

#[derive(Serialize)]
pub struct DevProfile {
    debug: bool,
}

#[derive(Serialize, Clone)]
pub struct Crate {
    pub path: Option<String>,
    pub version: String,
}

pub type Dependencies = HashMap<String, Crate>;

pub fn default_dependencies() -> Dependencies {
    [
        (
            "accel-core".to_string(),
            Crate {
                path: None,
                version: "*".to_string(),
            },
        ),
    ].iter()
        .cloned()
        .collect()
}

pub fn into_config(dependencies: Dependencies) -> CargoTOML {
    CargoTOML {
        package: Package {
            name: "ptx-builder".to_string(),
            version: "0.1.0".to_string(),
        },
        profile: Profile { dev: DevProfile { debug: false } },
        dependencies,
    }
}

impl CargoTOML {
    pub fn to_string(&self) -> String {
        toml::to_string(&self).unwrap()
    }
}
