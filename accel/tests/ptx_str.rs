use accel_derive::kernel;

#[kernel]
pub unsafe fn add(a: *const f64, b: *const f64, c: *mut f64, n: usize) {
    let i = accel_core::index();
    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}

#[test]
fn main() {
    // PTX assembler code is embedded as `add::PTX_STR`
    println!("{}", add::PTX_STR);
}
