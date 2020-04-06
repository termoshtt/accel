accel-derive
=============

[![Crate](http://meritbadge.herokuapp.com/accel-derive)](https://crates.io/crates/accel-derive)
[![docs.rs](https://docs.rs/accel-derive/badge.svg)](https://docs.rs/accel-derive)

Procedural-macro crate for `#[kernel]`. `#[kernel]` function will be converted to two part:

- Device code will be compiled into PTX assembler
- Host code which call the generated device code (PTX asm) using `accel::module` API
