# `yfft`

[<img src="https://docs.rs/yfft/badge.svg" alt="docs.rs">](https://docs.rs/yfft/)

Simple FFT library written purely in Rust. Requires a Nightly Rust compiler for x86 intrinsics.

![](docs/benchmark.jpg)

## Features

 - Features moderately optimized FFT kernels for small, power-of-two,
   single precision transforms.
 - Supports real-to-complex and complex-to-real transforms.
 - Clients can opt in to a swizzled input/output data order when they don't
   need naturally-ordered data.

## Limitations

This library was written in 2017 for internal use (specifically, real-time
game audio processing) and is not actively maintained anymore. For this
reason, this library has the following important limitations:

 - It's only optimized for small, power-of-two, single precision transforms.
   It may work for other sizes, but it will use extremely slow code paths.
 - It only supports 1D transforms.
 - It does not support detecting processor features at runtime.
 - The implementation relies on **a plenty of unsafe Rust code**.
   Use at your own risk!

## Notes Regarding Compilation

As of the version 1.19.0 cargo doesn't support passing codegen flags to rustc. Because of this,
you need to pass the following flags via the `RUSTFLAGS` environment variable to enable AVX kernel:

```sh
export RUSTFLAGS='-Ctarget-feature=+avx,+sse3'
```

Note: this causes codegen to generate VEX prefixes to all SSE instructions and makes the binary
incompatible with processors without AVX support.

## Example: Round-trip Conversion

```rust
use yfft::{Setup, Options, DataOrder, DataFormat, Env};

let size = 128;

let setup1: Setup<f32> = Setup::new(&Options {
    input_data_order: DataOrder::Natural,
    output_data_order: DataOrder::Swizzled,
    input_data_format: DataFormat::Complex,
    output_data_format: DataFormat::Complex,
    len: size,
    inverse: false,
})
.unwrap();
let setup2: Setup<f32> = Setup::new(&Options {
    input_data_order: DataOrder::Swizzled,
    output_data_order: DataOrder::Natural,
    input_data_format: DataFormat::Complex,
    output_data_format: DataFormat::Complex,
    len: size,
    inverse: true,
})
.unwrap();

// Allocate temporary buffers
let mut env1 = Env::new(&setup1);
let mut env2 = Env::new(&setup2);

// Input data (interleaved complex format)
let mut pat = vec![0.0f32; size * 2];
pat[42] = 100.0;
pat[82] = 200.0;

let mut result = vec![0.0f32; size * 2];
result.copy_from_slice(&pat);

// Round-trip transform
env1.transform(&mut result);
env2.transform(&mut result);

for e in &mut result {
    *e = *e / size as f32;
}

assert_num_slice_approx_eq(
    &result,
    &pat,
    1.0e-3f32
);

```

License: MIT/Apache-2.0
