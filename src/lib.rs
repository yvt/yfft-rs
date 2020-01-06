//! Simple FFT library written purely in Rust. Requires a Nightly Rust compiler for x86 intrinsics.
//!
//! Notes Regarding Compilation
//! ---------------------------
//!
//! As of the version 1.19.0 cargo doesn't support passing codegen flags to rustc. Because of this,
//! you need to pass the following flags via the `RUSTFLAGS` environemnt variable to enable AVX kernel:
//!
//! ```sh
//! export RUSTFLAGS='-Ctarget-feature=+avx,+sse3'
//! ```
//!
//! Note: this causes codegen to generate VEX prefixes to all SSE instructions and makes the binary
//! incompatibile with processors without AVX support.

#![cfg_attr(test, feature(test))]
#![feature(platform_intrinsics)]

extern crate num_complex;
extern crate num_iter;
extern crate num_traits;

#[macro_use]
extern crate packed_simd;

use std::fmt::Debug;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use num_complex::Complex;

#[macro_use]
mod simdutils;
mod aligned;
mod env;
mod kernel;
mod setup;

pub trait Num:
    Clone
    + Debug
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Default
    + num_traits::Float
    + num_traits::FloatConst
    + num_traits::Zero
    + 'static
    + Sync
    + Send
{
}
impl<T> Num for T where
    T: Clone
        + Debug
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Default
        + num_traits::Float
        + num_traits::FloatConst
        + num_traits::Zero
        + 'static
        + Sync
        + Send
{
}

#[inline]
fn complex_from_slice<T: Num>(x: &[T]) -> Complex<T> {
    Complex::new(x[0], x[1])
}

#[inline]
fn mul_pos_i<T: Num>(x: Complex<T>) -> Complex<T> {
    Complex::new(-x.im, x.re)
}

pub use env::Env;
pub use setup::{DataFormat, DataOrder, Options, PlanError, Setup};

#[cfg(test)]
mod benchmark;
