//! Defines Radix-4 FFT kernels optimized by using SSE instruction set.
//!
//! Performances
//! ------------
//!
//! According to a benchmark result, this kernel runs about 1-3x slower than a commercial-level FFT library (with
//! all optimizations and instruction sets including ones that this kernel doesn't support enabled) on a Skylake
//! machine.

use super::super::super::simdutils::{f32x4_bitxor, f32x4_complex_mul_rrii};
use super::utils::{
    branch_on_static_params, if_compatible, AlignInfo, AlignReqKernel, AlignReqKernelWrapper,
    StaticParams, StaticParamsConsumer,
};
use super::{Kernel, KernelCreationParams, KernelParams, KernelType, Num, SliceAccessor};

use num_complex::Complex;
use num_iter::range_step;

use packed_simd::f32x4;

use std::f32;

pub fn new_x86_sse_radix4_kernel<T>(cparams: &KernelCreationParams) -> Option<Box<Kernel<T>>>
where
    T: Num,
{
    if cparams.radix != 4 {
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
                SseRadix4Kernel3::new(cparams, sparams),
            ))),
            unit if unit % 2 == 0 => Some(Box::new(AlignReqKernelWrapper::new(
                SseRadix4Kernel2::new(cparams, sparams),
            ))),
            1 => Some(Box::new(AlignReqKernelWrapper::new(SseRadix4Kernel1::new(
                cparams, sparams,
            )))),
            _ => None,
        }
    }
}

/// This Radix-4 kernel is specialized for the case where `unit == 1` and computes one small FFTs in a single iteration.
#[derive(Debug)]
struct SseRadix4Kernel1<T> {
    cparams: KernelCreationParams,
    sparams: T,
}

impl<T: StaticParams> SseRadix4Kernel1<T> {
    fn new(cparams: &KernelCreationParams, sparams: T) -> Self {
        sparams.check_param(cparams);
        assert_eq!(cparams.radix, 4);
        assert_eq!(cparams.unit, 1);
        Self {
            cparams: *cparams,
            sparams: sparams,
        }
    }
}

impl<T: StaticParams> AlignReqKernel<f32> for SseRadix4Kernel1<T> {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<f32>) {
        let cparams = &self.cparams;
        let sparams = &self.sparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };

        let neg_mask_raw: [u32; 4] = if sparams.inverse() {
            [0, 0, 0x80000000, 0]
        } else {
            [0, 0, 0, 0x80000000]
        };
        let neg_mask = unsafe { *(&neg_mask_raw as *const u32 as *const f32x4) };

        for x in range_step(0, cparams.size * 2, 8) {
            let cur1 = &mut data[x] as *mut f32 as *mut f32x4;
            let cur2 = &mut data[x + 4] as *mut f32 as *mut f32x4;

            // riri format
            let x1 = unsafe { I::read(cur1) };
            let y1 = unsafe { I::read(cur2) };

            // perform size-4 small FFT (see generic2.rs for human-readable code)
            let t_1_2 = x1 + y1;
            let t_3_4 = x1 - y1;
            let t_1_3 = shuffle!(t_1_2, t_3_4, [0, 1, 4, 5]);
            let t_2_4t = shuffle!(t_1_2, t_3_4, [2, 3, 7, 6]);

            // multiply the last elem (t4) by I (backward) or -I (forward)
            let t_2_4 = f32x4_bitxor(t_2_4t, neg_mask);
            let x2 = t_1_3 + t_2_4;
            let y2 = t_1_3 - t_2_4;

            unsafe { I::write(cur1, x2) };
            unsafe { I::write(cur2, y2) };
        }
    }
    fn alignment_requirement(&self) -> usize {
        16
    }
}

/// This Radix-4 kernel computes two small FFTs in a single iteration.
#[derive(Debug)]
struct SseRadix4Kernel2<T> {
    cparams: KernelCreationParams,
    twiddles: Vec<f32x4>,
    sparams: T,
}

impl<T: StaticParams> SseRadix4Kernel2<T> {
    fn new(cparams: &KernelCreationParams, sparams: T) -> Self {
        sparams.check_param(cparams);
        assert_eq!(cparams.radix, 4);
        assert_eq!(cparams.unit % 2, 0);

        let full_circle = if cparams.inverse { 2f32 } else { -2f32 };
        let mut twiddles = Vec::new();
        for i in range_step(0, cparams.unit, 2) {
            let c1 = Complex::new(
                0f32,
                full_circle * (i) as f32 / (cparams.radix * cparams.unit) as f32 * f32::consts::PI,
            )
            .exp();
            let c2 = Complex::new(
                0f32,
                full_circle * (i + 1) as f32 / (cparams.radix * cparams.unit) as f32
                    * f32::consts::PI,
            )
            .exp();
            // rr-ii format
            twiddles.push(f32x4::new(c1.re, c2.re, c1.im, c2.im));

            let c12 = c1 * c1;
            let c22 = c2 * c2;
            twiddles.push(f32x4::new(c12.re, c22.re, c12.im, c22.im));

            let c13 = c12 * c1;
            let c23 = c22 * c2;
            twiddles.push(f32x4::new(c13.re, c23.re, c13.im, c23.im));
        }

        Self {
            cparams: *cparams,
            twiddles: twiddles,
            sparams: sparams,
        }
    }
}

impl<T: StaticParams> AlignReqKernel<f32> for SseRadix4Kernel2<T> {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<f32>) {
        let cparams = &self.cparams;
        let sparams = &self.sparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };

        let twiddles = unsafe { SliceAccessor::new(self.twiddles.as_slice()) };

        let neg_mask_raw: [u32; 4] = [0x80000000, 0x80000000, 0, 0];
        let neg_mask = unsafe { *(&neg_mask_raw as *const u32 as *const f32x4) };

        let neg_mask2_raw: [u32; 4] = [0x80000000, 0, 0x80000000, 0];
        let neg_mask2 = unsafe { *(&neg_mask2_raw as *const u32 as *const f32x4) };

        let pre_twiddle = sparams.kernel_type() == KernelType::Dit;
        let post_twiddle = sparams.kernel_type() == KernelType::Dif;

        for x in range_step(0, cparams.size * 2, cparams.unit * 8) {
            for y in 0..cparams.unit / 2 {
                let cur1 = &mut data[x + y * 4] as *mut f32 as *mut f32x4;
                let cur2 = &mut data[x + y * 4 + cparams.unit * 2] as *mut f32 as *mut f32x4;
                let cur3 = &mut data[x + y * 4 + cparams.unit * 4] as *mut f32 as *mut f32x4;
                let cur4 = &mut data[x + y * 4 + cparams.unit * 6] as *mut f32 as *mut f32x4;

                // rrii format
                let twiddle_1 = twiddles[y * 3];
                let twiddle_2 = twiddles[y * 3 + 1];
                let twiddle_3 = twiddles[y * 3 + 2];

                // riri format
                let x1 = unsafe { I::read(cur1) };
                let y1 = unsafe { I::read(cur2) };
                let z1 = unsafe { I::read(cur3) };
                let w1 = unsafe { I::read(cur4) };

                // apply twiddle factor
                let x2 = x1;
                let y2 = if pre_twiddle {
                    let t1 = shuffle!(y1, y1, [0, 2, 5, 7]); // riri to rrii
                    let t2 = f32x4_complex_mul_rrii(t1, twiddle_1, neg_mask);
                    shuffle!(t2, t2, [0, 2, 5, 7]) // rrii to riri
                } else {
                    y1
                };
                let z2 = if pre_twiddle {
                    let t1 = shuffle!(z1, z1, [0, 2, 5, 7]); // riri to rrii
                    let t2 = f32x4_complex_mul_rrii(t1, twiddle_2, neg_mask);
                    shuffle!(t2, t2, [0, 2, 5, 7]) // rrii to riri
                } else {
                    z1
                };
                let w2 = if pre_twiddle {
                    let t1 = shuffle!(w1, w1, [0, 2, 5, 7]); // riri to rrii
                    let t2 = f32x4_complex_mul_rrii(t1, twiddle_3, neg_mask);
                    shuffle!(t2, t2, [0, 2, 5, 7]) // rrii to riri
                } else {
                    w1
                };

                // perform size-4 FFT
                let x3 = x2 + z2;
                let y3 = y2 + w2;
                let z3 = x2 - z2;
                let w3t = y2 - w2;

                // w3 = w3t * i
                let w3 = f32x4_bitxor(shuffle!(w3t, w3t, [1, 0, 7, 6]), neg_mask2);

                let (x4, y4, z4, w4) = if sparams.inverse() {
                    (x3 + y3, z3 + w3, x3 - y3, z3 - w3)
                } else {
                    (x3 + y3, z3 - w3, x3 - y3, z3 + w3)
                };

                // apply twiddle factor
                let x5 = x4;
                let y5 = if post_twiddle {
                    let t1 = shuffle!(y4, y4, [0, 2, 5, 7]); // riri to rrii
                    let t2 = f32x4_complex_mul_rrii(t1, twiddle_1, neg_mask);
                    shuffle!(t2, t2, [0, 2, 5, 7]) // rrii to riri
                } else {
                    y4
                };
                let z5 = if post_twiddle {
                    let t1 = shuffle!(z4, z4, [0, 2, 5, 7]); // riri to rrii
                    let t2 = f32x4_complex_mul_rrii(t1, twiddle_2, neg_mask);
                    shuffle!(t2, t2, [0, 2, 5, 7]) // rrii to riri
                } else {
                    z4
                };
                let w5 = if post_twiddle {
                    let t1 = shuffle!(w4, w4, [0, 2, 5, 7]); // riri to rrii
                    let t2 = f32x4_complex_mul_rrii(t1, twiddle_3, neg_mask);
                    shuffle!(t2, t2, [0, 2, 5, 7]) // rrii to riri
                } else {
                    w4
                };

                unsafe { I::write(cur1, x5) };
                unsafe { I::write(cur2, y5) };
                unsafe { I::write(cur3, z5) };
                unsafe { I::write(cur4, w5) };
            }
        }
    }
    fn alignment_requirement(&self) -> usize {
        16
    }
}

/// This Radix-4 kernel computes four small FFTs in a single iteration.
#[derive(Debug)]
struct SseRadix4Kernel3<T: StaticParams> {
    cparams: KernelCreationParams,
    twiddles: Vec<f32x4>,
    sparams: T,
}

impl<T: StaticParams> SseRadix4Kernel3<T> {
    fn new(cparams: &KernelCreationParams, sparams: T) -> Self {
        sparams.check_param(cparams);
        assert_eq!(cparams.radix, 4);
        assert_eq!(cparams.unit % 4, 0);

        let full_circle = if cparams.inverse { 2f32 } else { -2f32 };
        let mut twiddles = Vec::new();
        for i in range_step(0, cparams.unit, 4) {
            let c1 = Complex::new(
                0f32,
                full_circle * (i) as f32 / (cparams.radix * cparams.unit) as f32 * f32::consts::PI,
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
            // rrrr-iiii format
            twiddles.push(f32x4::new(c1.re, c2.re, c3.re, c4.re));
            twiddles.push(f32x4::new(c1.im, c2.im, c3.im, c4.im));

            let c12 = c1 * c1;
            let c22 = c2 * c2;
            let c32 = c3 * c3;
            let c42 = c4 * c4;
            twiddles.push(f32x4::new(c12.re, c22.re, c32.re, c42.re));
            twiddles.push(f32x4::new(c12.im, c22.im, c32.im, c42.im));

            let c13 = c12 * c1;
            let c23 = c22 * c2;
            let c33 = c32 * c3;
            let c43 = c42 * c4;
            twiddles.push(f32x4::new(c13.re, c23.re, c33.re, c43.re));
            twiddles.push(f32x4::new(c13.im, c23.im, c33.im, c43.im));
        }

        Self {
            cparams: *cparams,
            twiddles: twiddles,
            sparams: sparams,
        }
    }
}

impl<T: StaticParams> AlignReqKernel<f32> for SseRadix4Kernel3<T> {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<f32>) {
        let cparams = &self.cparams;
        let sparams = &self.sparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };

        let twiddles = unsafe { SliceAccessor::new(self.twiddles.as_slice()) };
        let pre_twiddle = sparams.kernel_type() == KernelType::Dit;
        let post_twiddle = sparams.kernel_type() == KernelType::Dif;

        for x in range_step(0, cparams.size * 2, cparams.unit * 8) {
            for y in 0..cparams.unit / 4 {
                let cur1a = &mut data[x + y * 8] as *mut f32 as *mut f32x4;
                let cur1b = &mut data[x + y * 8 + 4] as *mut f32 as *mut f32x4;
                let cur2a = &mut data[x + y * 8 + cparams.unit * 2] as *mut f32 as *mut f32x4;
                let cur2b = &mut data[x + y * 8 + cparams.unit * 2 + 4] as *mut f32 as *mut f32x4;
                let cur3a = &mut data[x + y * 8 + cparams.unit * 4] as *mut f32 as *mut f32x4;
                let cur3b = &mut data[x + y * 8 + cparams.unit * 4 + 4] as *mut f32 as *mut f32x4;
                let cur4a = &mut data[x + y * 8 + cparams.unit * 6] as *mut f32 as *mut f32x4;
                let cur4b = &mut data[x + y * 8 + cparams.unit * 6 + 4] as *mut f32 as *mut f32x4;
                let twiddle1_r = twiddles[y * 6];
                let twiddle1_i = twiddles[y * 6 + 1];
                let twiddle2_r = twiddles[y * 6 + 2];
                let twiddle2_i = twiddles[y * 6 + 3];
                let twiddle3_r = twiddles[y * 6 + 4];
                let twiddle3_i = twiddles[y * 6 + 5];

                let x1a = unsafe { I::read(cur1a) };
                let x1b = unsafe { I::read(cur1b) };
                let y1a = unsafe { I::read(cur2a) };
                let y1b = unsafe { I::read(cur2b) };
                let z1a = unsafe { I::read(cur3a) };
                let z1b = unsafe { I::read(cur3b) };
                let w1a = unsafe { I::read(cur4a) };
                let w1b = unsafe { I::read(cur4b) };

                // convert riri-riri to rrrr-iiii (shufps)
                let x2r = shuffle!(x1a, x1b, [0, 2, 4, 6]);
                let x2i = shuffle!(x1a, x1b, [1, 3, 5, 7]);
                let y2r = shuffle!(y1a, y1b, [0, 2, 4, 6]);
                let y2i = shuffle!(y1a, y1b, [1, 3, 5, 7]);
                let z2r = shuffle!(z1a, z1b, [0, 2, 4, 6]);
                let z2i = shuffle!(z1a, z1b, [1, 3, 5, 7]);
                let w2r = shuffle!(w1a, w1b, [0, 2, 4, 6]);
                let w2i = shuffle!(w1a, w1b, [1, 3, 5, 7]);

                // apply twiddle factor
                let x3r = x2r;
                let x3i = x2i;
                let y3r = if pre_twiddle {
                    y2r * twiddle1_r - y2i * twiddle1_i
                } else {
                    y2r
                };
                let y3i = if pre_twiddle {
                    y2r * twiddle1_i + y2i * twiddle1_r
                } else {
                    y2i
                };
                let z3r = if pre_twiddle {
                    z2r * twiddle2_r - z2i * twiddle2_i
                } else {
                    z2r
                };
                let z3i = if pre_twiddle {
                    z2r * twiddle2_i + z2i * twiddle2_r
                } else {
                    z2i
                };
                let w3r = if pre_twiddle {
                    w2r * twiddle3_r - w2i * twiddle3_i
                } else {
                    w2r
                };
                let w3i = if pre_twiddle {
                    w2r * twiddle3_i + w2i * twiddle3_r
                } else {
                    w2i
                };

                // perform size-4 FFT
                let x4r = x3r + z3r;
                let x4i = x3i + z3i;
                let y4r = y3r + w3r;
                let y4i = y3i + w3i;
                let z4r = x3r - z3r;
                let z4i = x3i - z3i;
                let w4r = y3r - w3r;
                let w4i = y3i - w3i;

                let x5r = x4r + y4r;
                let x5i = x4i + y4i;
                let z5r = x4r - y4r;
                let z5i = x4i - y4i;
                let (y5r, y5i, w5r, w5i) = if self.sparams.inverse() {
                    (z4r - w4i, z4i + w4r, z4r + w4i, z4i - w4r)
                } else {
                    (z4r + w4i, z4i - w4r, z4r - w4i, z4i + w4r)
                };

                // apply twiddle factor
                let x6r = x5r;
                let x6i = x5i;
                let y6r = if post_twiddle {
                    y5r * twiddle1_r - y5i * twiddle1_i
                } else {
                    y5r
                };
                let y6i = if post_twiddle {
                    y5r * twiddle1_i + y5i * twiddle1_r
                } else {
                    y5i
                };
                let z6r = if post_twiddle {
                    z5r * twiddle2_r - z5i * twiddle2_i
                } else {
                    z5r
                };
                let z6i = if post_twiddle {
                    z5r * twiddle2_i + z5i * twiddle2_r
                } else {
                    z5i
                };
                let w6r = if post_twiddle {
                    w5r * twiddle3_r - w5i * twiddle3_i
                } else {
                    w5r
                };
                let w6i = if post_twiddle {
                    w5r * twiddle3_i + w5i * twiddle3_r
                } else {
                    w5i
                };

                // convert to rrrr-iiii to riri-riri (unpcklps/unpckups)
                let x7a = shuffle!(x6r, x6i, [0, 4, 1, 5]);
                let x7b = shuffle!(x6r, x6i, [2, 6, 3, 7]);
                let y7a = shuffle!(y6r, y6i, [0, 4, 1, 5]);
                let y7b = shuffle!(y6r, y6i, [2, 6, 3, 7]);
                let z7a = shuffle!(z6r, z6i, [0, 4, 1, 5]);
                let z7b = shuffle!(z6r, z6i, [2, 6, 3, 7]);
                let w7a = shuffle!(w6r, w6i, [0, 4, 1, 5]);
                let w7b = shuffle!(w6r, w6i, [2, 6, 3, 7]);

                unsafe { I::write(cur1a, x7a) };
                unsafe { I::write(cur1b, x7b) };
                unsafe { I::write(cur2a, y7a) };
                unsafe { I::write(cur2b, y7b) };
                unsafe { I::write(cur3a, z7a) };
                unsafe { I::write(cur3b, z7b) };
                unsafe { I::write(cur4a, w7a) };
                unsafe { I::write(cur4b, w7b) };
            }
        }
    }
    fn alignment_requirement(&self) -> usize {
        16
    }
}
