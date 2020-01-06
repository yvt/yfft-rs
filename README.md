# `yfft`

[<img src="https://docs.rs/yfft/badge.svg" alt="docs.rs">](https://docs.rs/yfft/)

Simple FFT library written purely in Rust. Requires a Nightly Rust compiler for x86 intrinsics.

Notes Regarding Compilation
---------------------------

As of the version 1.19.0 cargo doesn't support passing codegen flags to rustc. Because of this,
you need to pass the following flags via the `RUSTFLAGS` environemnt variable to enable AVX kernel:

```sh
export RUSTFLAGS='-Ctarget-feature=+avx,+sse3'
```

Note: this causes codegen to generate VEX prefixes to all SSE instructions and makes the binary
incompatibile with processors without AVX support.

License: MIT/Apache-2.0
