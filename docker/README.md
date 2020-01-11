rust-cuda containers
=====================

Docker container including

- CUDA based on [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/) containers
- NVPTX target for Rust

```
docker run -it --rm registry.gitlab.com/termoshtt/accel/ubuntu18.04-cuda10.2:master
```

See also https://gitlab.com/termoshtt/accel/container_registry

Supported Platforms
------------------

|CUDA | Ubuntu 18.04 | Ubuntu 16.04 | RedHat UBI8 | RedHat UBI7 | CentOS 7 | CentOS 6 |
|:---:|:------------:|:------------:|:-----------:|:-----------:|:--------:|:--------:|
|10.2 | ✔️            | ✔️            |             |             | ✔️        | ✔️        |
|10.1 | ✔️            | ✔️            |             |             | ✔️        | ✔️        |
|10.0 | ✔️            | ✔️            | -           | -           | ✔️        | ✔️        |
|9.2  | ✔️            | ✔️            | -           | -           | ✔️        | ✔️        |
|9.1  | -            | ✔️            | -           | -           | ✔️        | ✔️        |
|9.0  | -            | ✔️            | -           | -           | ✔️        | ✔️        |
|8.0  | -            | ✔️            | -           | -           | ✔️        | ✔️        |

- https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
- https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/unsupported-tags.md
