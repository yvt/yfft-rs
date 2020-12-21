mod accessor;
mod bitreversal;
mod convert;
mod generic;
mod generic2;
mod realfft;
mod utils;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

// Stub for non-x86 systems
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
mod x86 {
    pub fn new_x86_kernel<T>(
        cparams: &super::KernelCreationParams,
    ) -> Option<Box<super::Kernel<T>>> {
        None
    }
    pub unsafe fn new_x86_bit_reversal_kernel<T>(
        indices: &Vec<usize>,
    ) -> Option<Box<super::Kernel<T>>> {
        None
    }
    pub fn new_x86_real_fft_pre_post_process_kernel<T>(
        len: usize,
        inverse: bool,
    ) -> Option<Box<super::Kernel<T>>> {
        None
    }
}

use super::Num;
use std::fmt::Debug;

use self::accessor::SliceAccessor;

pub use self::bitreversal::new_bit_reversal_kernel;
pub use self::convert::*;
pub use self::realfft::*;

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum KernelType {
    /// Decimation-in-time.
    Dit,

    /// Decimation-in-frequency.
    Dif,
}

// for Radix-2 DIT, (dim1, dim2) = (2, x)
// for Radix-2 DIF, (dim1, dim2) = (x, 2) where x <= size / 2
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct KernelCreationParams {
    pub size: usize,
    pub kernel_type: KernelType,
    pub radix: usize,

    /// It's kinda hard to describe so I'll just put a bound here:
    /// `1 <= unit <= size / radix` I hope you get the idea.
    pub unit: usize,

    pub inverse: bool,
}

#[derive(Debug)]
pub struct KernelParams<'a, T: 'a> {
    pub coefs: &'a mut [T],
    pub work_area: &'a mut [T],
}

pub trait Kernel<T>: Debug + Sync + Send {
    fn transform(&self, params: &mut KernelParams<T>);
    fn required_work_area_size(&self) -> usize {
        0
    }
}

impl<T> Kernel<T>
where
    T: Num + 'static,
{
    pub fn new(cparams: &KernelCreationParams) -> Box<Kernel<T>> {
        x86::new_x86_kernel(cparams)
            .or_else(|| generic2::new_specialized_generic_kernel(cparams))
            .unwrap_or_else(|| generic::new_generic_kernel(cparams))
    }
}
