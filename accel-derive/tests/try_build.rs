#[test]
fn kernel_generate() {
    let t = trybuild::TestCases::new();
    t.pass("tests/kernels/do_nothing.rs");
    t.pass("tests/kernels/dependencies.rs");
    t.pass("tests/kernels/dependencies_git.rs");
}
