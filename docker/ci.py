#!/usr/bin/env python3

import yaml


CUDA_ALL_VERSIONS = ["8.0", "9.0", "9.1", "9.2", "10.0", "10.1", "10.2"]
CUDA_RECENT_VERSIONS = ["9.2", "10.0", "10.1", "10.2"]

UBUNTU = {"16.04": CUDA_ALL_VERSIONS, "18.04": CUDA_RECENT_VERSIONS}
CENTOS = {"6": CUDA_ALL_VERSIONS, "7": CUDA_ALL_VERSIONS}


def main():
    ci_config = {}
    for ubuntu_version, cuda_versions in UBUNTU.items():
        for cuda_version in cuda_versions:
            tag = f"ubuntu{ubuntu_version}-cuda{cuda_version}"
            ci_config[tag] = {"extends": ".build", "variables": {"TAG": tag}}
    for centos_version, cuda_versions in CENTOS.items():
        for cuda_version in cuda_versions:
            tag = f"centos{centos_version}-cuda{cuda_version}"
            ci_config[tag] = {"extends": ".build", "variables": {"TAG": tag}}

    print(yaml.dump(ci_config))


if __name__ == "__main__":
    main()
