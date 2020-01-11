use accel_derive::kernel;

#[kernel]
#[dependencies("accel-core" = { git = "https://gitlab.com/termoshtt/accel" })]
unsafe fn git() {
    let _i = accel_core::index();
}

#[kernel]
#[dependencies("accel-core" = { git = "https://gitlab.com/termoshtt/accel", branch = "master" })]
unsafe fn git_branch() {
    let _i = accel_core::index();
}

fn main() {}
