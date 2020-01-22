//! Defines generic FFT kernels that don't make any assumptions about radix or instruction set.
//!
//! Performances
//! ------------
//!
//! According to a benchmark result, this kernel runs about 100x slower than a commercial-level FFT library on a Skylake
//! machine.

use super::{Kernel, KernelCreationParams, KernelParams, KernelType, SliceAccessor};

use num_complex::Complex;
use num_iter::range_step;
use num_traits::{One, Zero};

use super::super::{complex_from_slice, Num};

pub fn new_generic_kernel<T: 'static>(cparams: &KernelCreationParams) -> Box<Kernel<T>>
where
    T: Num,
{
    let full_circle = if cparams.inverse { 2 } else { -2 };
    let twiddle_delta = Complex::new(
        Zero::zero(),
        T::from(cparams.size / cparams.radix / cparams.unit).unwrap()
            * T::from(full_circle).unwrap()
            * T::PI()
            / T::from(cparams.size).unwrap(),
    )
    .exp();
    let coef_delta = Complex::new(
        Zero::zero(),
        T::from(full_circle).unwrap() * T::PI() / T::from(cparams.radix).unwrap(),
    )
    .exp();

    match cparams.kernel_type {
        KernelType::Dit => Box::new(GenericDitKernel {
            cparams: *cparams,
            twiddle_delta: twiddle_delta,
            coef_delta: coef_delta,
        }),
        KernelType::Dif => Box::new(GenericDifKernel {
            cparams: *cparams,
            twiddle_delta: twiddle_delta,
            coef_delta: coef_delta,
        }),
    }
}

#[derive(Debug)]
struct GenericDitKernel<T> {
    cparams: KernelCreationParams,
    twiddle_delta: Complex<T>,
    /// sub-FFT twiddle unit
    coef_delta: Complex<T>,
}

#[derive(Debug)]
struct GenericDifKernel<T> {
    cparams: KernelCreationParams,
    twiddle_delta: Complex<T>,
    /// sub-FFT twiddle unit
    coef_delta: Complex<T>,
}

impl<T> Kernel<T> for GenericDitKernel<T>
where
    T: Num,
{
    fn transform(&self, params: &mut KernelParams<T>) {
        let cparams = &self.cparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };
        let mut wa = unsafe { SliceAccessor::new(&mut params.work_area[0..cparams.radix * 2]) };

        let twiddle_delta = self.twiddle_delta;
        let coef_delta = self.coef_delta;

        for x in range_step(0, cparams.size, cparams.unit * cparams.radix) {
            let mut twiddle_1: Complex<T> = Complex::one();
            for y in 0..cparams.unit {
                for z in 0..cparams.radix {
                    wa[z * 2] = data[(x + y + z * cparams.unit) * 2];
                    wa[z * 2 + 1] = data[(x + y + z * cparams.unit) * 2 + 1];
                }
                let mut coef_1: Complex<T> = Complex::one();
                for z in 0..cparams.radix {
                    let mut c: Complex<T> = Complex::zero();
                    let mut coef_2: Complex<T> = Complex::one();
                    let coef_1_tw: Complex<T> = coef_1 * twiddle_1;
                    for w in 0..cparams.radix {
                        c = c + coef_2 * complex_from_slice(&wa[w * 2..]);
                        coef_2 = coef_2 * coef_1_tw;
                    }
                    data[(x + y + z * cparams.unit) * 2] = c.re;
                    data[(x + y + z * cparams.unit) * 2 + 1] = c.im;
                    coef_1 = coef_1 * coef_delta;
                }
                twiddle_1 = twiddle_1 * twiddle_delta;
            }
        }
    }

    fn required_work_area_size(&self) -> usize {
        self.cparams.radix * 2
    }
}

impl<T> Kernel<T> for GenericDifKernel<T>
where
    T: Num,
{
    fn transform(&self, params: &mut KernelParams<T>) {
        let cparams = &self.cparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };
        let mut wa = unsafe { SliceAccessor::new(&mut params.work_area[0..cparams.radix * 2]) };

        let twiddle_delta = self.twiddle_delta;
        let coef_delta = self.coef_delta;

        for x in range_step(0, cparams.size, cparams.unit * cparams.radix) {
            let mut twiddle_1: Complex<T> = Complex::one();
            for y in 0..cparams.unit {
                for z in 0..cparams.radix {
                    wa[z * 2] = data[(x + y + z * cparams.unit) * 2];
                    wa[z * 2 + 1] = data[(x + y + z * cparams.unit) * 2 + 1];
                }
                let mut twiddle_2: Complex<T> = Complex::one();
                let mut coef_1: Complex<T> = Complex::one();
                for z in 0..cparams.radix {
                    let mut c: Complex<T> = Complex::zero();
                    let mut coef_2 = twiddle_2;
                    for w in 0..cparams.radix {
                        c = c + coef_2 * complex_from_slice(&wa[w * 2..]);
                        coef_2 = coef_2 * coef_1;
                    }
                    data[(x + y + z * cparams.unit) * 2] = c.re;
                    data[(x + y + z * cparams.unit) * 2 + 1] = c.im;
                    twiddle_2 = twiddle_2 * twiddle_1;
                    coef_1 = coef_1 * coef_delta;
                }
                twiddle_1 = twiddle_1 * twiddle_delta;
            }
        }
    }

    fn required_work_area_size(&self) -> usize {
        self.cparams.radix * 2
    }
}
