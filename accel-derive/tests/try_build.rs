#[test]
fn kernel_generate() {
    let t = trybuild::TestCases::new();
    t.pass("examples/do_nothing.rs");
    t.pass("examples/dependencies.rs");
}
