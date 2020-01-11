#[test]
fn do_nothing() {
    let t = trybuild::TestCases::new();
    t.pass("tests/do_nothing.rs");
}
