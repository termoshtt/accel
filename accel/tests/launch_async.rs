#[test]
fn launch_async_build_test() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/launch_async/mut_ref.rs");
}
