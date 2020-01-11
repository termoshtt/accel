#[test]
fn do_nothing() {
    let t = trybuild::TestCases::new();
    t.pass("test_kernels/do_nothing.rs");
}

#[test]
fn dependencies() {
    let t = trybuild::TestCases::new();
    t.pass("test_kernels/dependencies.rs");
}
