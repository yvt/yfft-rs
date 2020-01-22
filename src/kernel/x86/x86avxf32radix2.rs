//! Defines Radix-2 single precision FFT kernels optimized by using AVX instruction set.
//!
//! AVX expands the register width to 256bit and adds the 256-bit counterparts of most existing instructions.
//!
//! Performances
//! ------------
//!
//! For small transforms ties with a commercial-level FFT library, but tends to be much slower for large transforms.

use super::super::super::simdutils::{avx_f32x8_bitxor, avx_f32x8_complex_mul_riri};
use super::utils::{
    branch_on_static_params, if_compatible, AlignInfo, AlignReqKernel, AlignReqKernelWrapper,
    StaticParams, StaticParamsConsumer,
};
use super::{Kernel, KernelCreationParams, KernelParams, KernelType, Num, SliceAccessor};

use num_complex::Complex;
use num_iter::range_step;

use packed_simd::{f32x8, u32x8};

use std::{f32, mem};

pub fn new_x86_avx_f32_radix2_kernel<T>(cparams: &KernelCreationParams) -> Option<Box<Kernel<T>>>
where
    T: Num,
{
    if cparams.radix != 2 {
        return None;
    }

    if_compatible(|| branch_on_static_params(cparams, Factory {}))
}

struct Factory {}
impl StaticParamsConsumer<Option<Box<Kernel<f32>>>> for Factory {
    fn consume<T>(self, cparams: &KernelCreationParams, sparams: T) -> Option<Box<Kernel<f32>>>
    where
        T: StaticParams,
    {
        match cparams.unit {
            unit if unit % 4 == 0 => Some(Box::new(AlignReqKernelWrapper::new(
                AvxRadix2Kernel2::new(cparams, sparams),
            ))),
            1 if cparams.size % 4 == 0 => {
                Some(Box::new(AlignReqKernelWrapper::new(AvxRadix2Kernel1 {
                    cparams: *cparams,
                })))
            }
            _ => None,
        }
    }
}

/// This Radix-2 kernel is specialized for the case where `unit == 1` and computes two small FFTs in a single iteration.
#[derive(Debug)]
struct AvxRadix2Kernel1 {
    cparams: KernelCreationParams,
}

impl AlignReqKernel<f32> for AvxRadix2Kernel1 {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<f32>) {
        let cparams = &self.cparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };

        assert_eq!(cparams.radix, 2);
        assert_eq!(cparams.unit, 1);
        assert_eq!(cparams.size % 4, 0);

        let neg_mask: f32x8 = unsafe {
            mem::transmute(u32x8::new(
                0, 0, 0x80000000, 0x80000000, 0, 0, 0x80000000, 0x80000000,
            ))
        };

        for x in range_step(0, cparams.size * 2, 8) {
            let cur = &mut data[x] as *mut f32 as *mut f32x8;
            // t1a, t1b : Complex<f32> = X[x/2 .. x/2 + 2]
            let t1 = unsafe { I::read(cur) };
            // t2a, t2b = t1b, t1a
            let t2 = shuffle!(t1, t1, [2, 3, 8, 9, 6, 7, 12, 13]);
            // t3a, t3b = t1a, -t1b
            let t3 = avx_f32x8_bitxor(t1, neg_mask);
            // t4a, t4b = t2a + t3a, t3b + t3b = t1a + t1b, t1a - t1b
            let t4 = t2 + t3;
            // Y[x/2 .. x/2 + 2] = t4a, t4b
            unsafe { I::write(cur, t4) };
        }
    }
    fn alignment_requirement(&self) -> usize {
        32
    }
}

/// This Radix-2 kernel computes eight small FFTs in a single iteration.
#[derive(Debug)]
struct AvxRadix2Kernel2<T> {
    cparams: KernelCreationParams,
    twiddles: Vec<f32x8>,
    sparams: T,
}

impl<T: StaticParams> AvxRadix2Kernel2<T> {
    fn new(cparams: &KernelCreationParams, sparams: T) -> Self {
        sparams.check_param(cparams);
        assert_eq!(cparams.radix, 2);
        assert_eq!(cparams.unit % 4, 0);

        let full_circle = if cparams.inverse { 2f32 } else { -2f32 };
        let twiddles = range_step(0, cparams.unit, 4)
            .map(|i| {
                let c1 = Complex::new(
                    0f32,
                    full_circle * (i) as f32 / (cparams.radix * cparams.unit) as f32
                        * f32::consts::PI,
                )
                .exp();
                let c2 = Complex::new(
                    0f32,
                    full_circle * (i + 1) as f32 / (cparams.radix * cparams.unit) as f32
                        * f32::consts::PI,
                )
                .exp();
                let c3 = Complex::new(
                    0f32,
                    full_circle * (i + 2) as f32 / (cparams.radix * cparams.unit) as f32
                        * f32::consts::PI,
                )
                .exp();
                let c4 = Complex::new(
                    0f32,
                    full_circle * (i + 3) as f32 / (cparams.radix * cparams.unit) as f32
                        * f32::consts::PI,
                )
                .exp();
                // riririri format
                f32x8::new(c1.re, c1.im, c2.re, c2.im, c3.re, c3.im, c4.re, c4.im)
            })
            .collect();

        Self {
            cparams: *cparams,
            twiddles: twiddles,
            sparams: sparams,
        }
    }
}

impl<T: StaticParams> AlignReqKernel<f32> for AvxRadix2Kernel2<T> {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<f32>) {
        let cparams = &self.cparams;
        let sparams = &self.sparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };

        let twiddles = unsafe { SliceAccessor::new(self.twiddles.as_slice()) };

        let pre_twiddle = sparams.kernel_type() == KernelType::Dit;
        let post_twiddle = sparams.kernel_type() == KernelType::Dif;

        for x in range_step(0, cparams.size * 2, cparams.unit * 4) {
            for y in 0..cparams.unit / 4 {
                let cur1 = &mut data[x + y * 8] as *mut f32 as *mut f32x8;
                let cur2 = &mut data[x + y * 8 + cparams.unit * 2] as *mut f32 as *mut f32x8;
                let twiddle_1 = twiddles[y];

                // x1a, x1b : Complex<f32> = X[x1/2 .. x1/2 + 2]
                // (x1a.r, x1a.i, x1b.r, x1b.i, ...)
                let x1 = unsafe { I::read(cur1) };
                // y1a, y1b : Complex<f32> = X[x2/2 .. x2/2 + 2]
                // (y1a.r, y1a.i, y1b.r, y1b.i, ...)
                let y1 = unsafe { I::read(cur2) };

                // apply twiddle factor
                // (y1a.r, y1b.r, y1a.i, y1b.i)
                let x2 = x1;
                let y2 = if pre_twiddle {
                    avx_f32x8_complex_mul_riri(y1, twiddle_1)
                } else {
                    y1
                };

                // perform size-2 FFT
                // (y3a.r, y3a.i, y3b.r, y3b.i)
                let x3 = x2 + y2;
                let y3 = x2 - y2;

                // apply twiddle factor
                // (y1a.r, y1b.r, y1a.i, y1b.i)
                let x4 = x3;
                let y4 = if post_twiddle {
                    avx_f32x8_complex_mul_riri(y3, twiddle_1)
                } else {
                    y3
                };

                unsafe { I::write(cur1, x4) };
                unsafe { I::write(cur2, y4) };
            }
        }
    }
    fn alignment_requirement(&self) -> usize {
        32
    }
}
