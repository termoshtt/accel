use accel_derive::kernel;

#[kernel]
#[dependencies("accel-core" = { git = "https://gitlab.com/termoshtt/accel" })]
unsafe fn git() {}

#[kernel]
#[dependencies("accel-core" = { git = "https://gitlab.com/termoshtt/accel", branch = "master" })]
unsafe fn git_branch() {}

fn main() {}
