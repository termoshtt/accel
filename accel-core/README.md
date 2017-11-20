accel-core
===========

[![Crate](http://meritbadge.herokuapp.com/accel-core)](https://crates.io/crates/accel-core)
[![docs.rs](https://docs.rs/accel-core/badge.svg)](https://docs.rs/accel-core)

Support crate for writing kernels

- This crate will automatically be inserted by [ptx_builder](../src/ptx_builder) into the kernel crate.
- If `$ACCEL_HOME` environmental value exists, `$ACCEL_HOME/accel-core` is used. This is useful for debug.
