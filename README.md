# acc

Plan to develop OpenACC-like extension for Rust:

```rust
extern crate acc;

#[kernel(i)]
fn add(a: &[f64], b: &[f64], c: &mut [f64]) {
  *c[i] = a[i] + b[i];
}

fn main() {
  let N = 1000_000;
  let a = vec![0.0; N];
  let b = vec![1.0; N];
  let mut c = vec![0.0; N];
  add(&a, &b, &mut c);
}
```

Code is for illustration purposes (/・ω・)/
