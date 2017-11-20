accel-derive
=============

[![Crate](http://meritbadge.herokuapp.com/accel-derive)](https://crates.io/crates/accel-derive)
[![docs.rs](https://docs.rs/accel-derive/badge.svg)](https://docs.rs/accel-derive)

This is `proc-macro` crate to derive `#[kernel]`.

Stablize
---------

This crate utlizes [Macro 2.0](https://github.com/rust-lang/rust/issues/39412) througth crates developed in [futures-await](https://github.com/alexcrichton/futures-await) crate:

- [proc-macro2](https://github.com/alexcrichton/proc-macro2)
- [futures-await-syn](https://github.com/alexcrichton/futures-await-syn)
- [futures-await-quote](https://github.com/alexcrichton/futures-await-quote)
