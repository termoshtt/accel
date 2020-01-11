#[test]
fn do_nothing() {
    let t = trybuild::TestCases::new();
    t.pass("examples/do_nothing.rs");
}

#[test]
fn dependencies() {
    let t = trybuild::TestCases::new();
    t.pass("examples/dependencies.rs");
}
