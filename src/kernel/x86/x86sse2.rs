
//! Defines FFT kernels optimized by using SSE2 instruction set.
//!
//! Performances
//! ------------
//!
//! Yet to be measured.

use super::{Kernel, KernelCreationParams, Num};

pub fn new_x86_sse2_kernel<T>(_: &KernelCreationParams) -> Option<Box<Kernel<T>>>
where
    T: Num,
{
    None
}
