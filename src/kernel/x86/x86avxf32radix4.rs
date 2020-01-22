//! Defines Radix-4 single precision FFT kernels optimized by using AVX instruction set.
//!
//! AVX expands the register width to 256bit and adds the 256-bit counterparts of most existing instructions.
//!
//! Performances
//! ------------
//!
//! For small transforms ties with a commercial-level FFT library, but tends to be much slower for large transforms.

use super::utils::{
    branch_on_static_params, if_compatible, AlignInfo, AlignReqKernel, AlignReqKernelWrapper,
    StaticParams, StaticParamsConsumer,
};
use super::{Kernel, KernelCreationParams, KernelParams, KernelType, Num, SliceAccessor};
use simdutils::{
    avx_f32x8_bitxor, avx_f32x8_complex_mul_riri, avx_fma_f32x8_fmadd, avx_fma_f32x8_fmsub,
};

use num_complex::Complex;
use num_iter::range_step;

use packed_simd::{f32x8, u32x8};

use std::{f32, mem};

pub fn new_x86_avx_f32_radix4_kernel<T>(cparams: &KernelCreationParams) -> Option<Box<Kernel<T>>>
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
            // heuristics
            unit if unit % 8 == 0 && cparams.size <= 2048 => Some(Box::new(
                AlignReqKernelWrapper::new(AvxRadix4Kernel4::new(cparams, sparams)),
            )),
            unit if unit % 4 == 0 => Some(Box::new(AlignReqKernelWrapper::new(
                AvxRadix4Kernel3::new(cparams, sparams),
            ))),
            2 => Some(Box::new(AlignReqKernelWrapper::new(AvxRadix4Kernel2::new(
                cparams, sparams,
            )))),
            _ => None,
        }
    }
}

/// This Radix-4 kernel computes two small FFTs in a single iteration. Specialized for `unit == 2`.
#[derive(Debug)]
struct AvxRadix4Kernel2<T> {
    cparams: KernelCreationParams,
    twiddles: f32x8,
    sparams: T,
}

impl<T: StaticParams> AvxRadix4Kernel2<T> {
    fn new(cparams: &KernelCreationParams, sparams: T) -> Self {
        sparams.check_param(cparams);
        assert_eq!(cparams.radix, 4);
        assert_eq!(cparams.unit, 2);

        let full_circle = if cparams.inverse { 2f32 } else { -2f32 };
        let c1 = Complex::new(
            0f32,
            full_circle * 2 as f32 / (cparams.radix * cparams.unit) as f32 * f32::consts::PI,
        )
        .exp();
        let c2 = Complex::new(
            0f32,
            full_circle * 1 as f32 / (cparams.radix * cparams.unit) as f32 * f32::consts::PI,
        )
        .exp();
        let c3 = Complex::new(
            0f32,
            full_circle * 3 as f32 / (cparams.radix * cparams.unit) as f32 * f32::consts::PI,
        )
        .exp();
        // riri format
        let twiddles = f32x8::new(1f32, 0f32, c1.re, c1.im, c2.re, c2.im, c3.re, c3.im);

        Self {
            cparams: *cparams,
            twiddles: twiddles,
            sparams: sparams,
        }
    }
}

impl<T: StaticParams> AlignReqKernel<f32> for AvxRadix4Kernel2<T> {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<f32>) {
        let cparams = &self.cparams;
        let sparams = &self.sparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };

        let twiddles = self.twiddles;

        let neg_mask2: f32x8 = unsafe {
            mem::transmute(if sparams.inverse() {
                u32x8::new(0, 0, 0, 0, 0x80000000, 0, 0x80000000, 0)
            } else {
                u32x8::new(0, 0, 0, 0, 0, 0x80000000, 0, 0x80000000)
            })
        };

        let pre_twiddle = sparams.kernel_type() == KernelType::Dit;
        let post_twiddle = sparams.kernel_type() == KernelType::Dif;

        for x in range_step(0, cparams.size * 2, 16) {
            let cur1 = &mut data[x] as *mut f32 as *mut f32x8;
            let cur2 = &mut data[x + 8] as *mut f32 as *mut f32x8;

            // riri format
            let xy1 = unsafe { I::read(cur1) };
            let zw1 = unsafe { I::read(cur2) };

            // apply twiddle factor
            let (xy2, zw2) = if pre_twiddle {
                // riririri-riririri -> riririri
                //   12  34   56  78    12563478
                let t1: f32x8 = shuffle!(xy1, zw1, [2, 3, 10, 11, 6, 7, 14, 15]);
                let t2: f32x8 = avx_f32x8_complex_mul_riri(t1, twiddles);
                // t3: --12--34 (vmovddup)
                let t3: f32x8 = shuffle!(t2, t2, [0, 1, 8, 9, 4, 5, 12, 13]);
                // vblendps
                (
                    shuffle!(xy1, t3, [0, 1, 10, 11, 4, 5, 14, 15]),
                    shuffle!(zw1, t2, [0, 1, 10, 11, 4, 5, 14, 15]),
                )
            } else {
                (xy1, zw1)
            };

            // perform size-4 FFT
            let t12 = xy2 + zw2;
            let t34 = xy2 - zw2;

            // transpose (vperm2f128)
            let t13: f32x8 = shuffle!(t12, t34, [0, 1, 2, 3, 8, 9, 10, 11]);
            let t24t: f32x8 = shuffle!(t12, t34, [4, 5, 6, 7, 12, 13, 14, 15]);

            // t4 = t4 * i (backward), t4 = t4 * -i (forward)
            let t24t2: f32x8 = shuffle!(t24t, t24t, [1, 0, 3, 2, 5, 4, 7, 6]); // vpermilps (t3 * i, t4 * i)
            let t24t3: f32x8 = shuffle!(t24t, t24t2, [0, 1, 2, 3, 12, 13, 14, 15]); // vblendps or vperm2f128 (t3, t4 * i)
            let t24: f32x8 = avx_f32x8_bitxor(t24t3, neg_mask2);

            let (xy3, zw3) = (t13 + t24, t13 - t24);

            // apply twiddle factor
            let (xy4, zw4) = if post_twiddle {
                // riririri-riririri -> riririri
                //   12  34   56  78    12563478
                let t1: f32x8 = shuffle!(xy3, zw3, [2, 3, 10, 11, 6, 7, 14, 15]);
                let t2: f32x8 = avx_f32x8_complex_mul_riri(t1, twiddles);
                // t3: --12--34 (vmovddup)
                let t3: f32x8 = shuffle!(t2, t2, [0, 1, 8, 9, 4, 5, 12, 13]);
                // vblendps
                (
                    shuffle!(xy3, t3, [0, 1, 10, 11, 4, 5, 14, 15]),
                    shuffle!(zw3, t2, [0, 1, 10, 11, 4, 5, 14, 15]),
                )
            } else {
                (xy3, zw3)
            };

            unsafe { I::write(cur1, xy4) };
            unsafe { I::write(cur2, zw4) };
        }
    }
    fn alignment_requirement(&self) -> usize {
        32
    }
}

/// This Radix-4 kernel computes four small FFTs in a single iteration.
#[derive(Debug)]
struct AvxRadix4Kernel3<T> {
    cparams: KernelCreationParams,
    twiddles: Vec<f32x8>,
    sparams: T,
}

impl<T: StaticParams> AvxRadix4Kernel3<T> {
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
            // riri format
            twiddles.push(f32x8::new(
                c1.re, c1.im, c2.re, c2.im, c3.re, c3.im, c4.re, c4.im,
            ));

            let c12 = c1 * c1;
            let c22 = c2 * c2;
            let c32 = c3 * c3;
            let c42 = c4 * c4;
            twiddles.push(f32x8::new(
                c12.re, c12.im, c22.re, c22.im, c32.re, c32.im, c42.re, c42.im,
            ));

            let c13 = c12 * c1;
            let c23 = c22 * c2;
            let c33 = c32 * c3;
            let c43 = c42 * c4;
            twiddles.push(f32x8::new(
                c13.re, c13.im, c23.re, c23.im, c33.re, c33.im, c43.re, c43.im,
            ));
        }

        Self {
            cparams: *cparams,
            twiddles: twiddles,
            sparams: sparams,
        }
    }
}

impl<T: StaticParams> AlignReqKernel<f32> for AvxRadix4Kernel3<T> {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<f32>) {
        let cparams = &self.cparams;
        let sparams = &self.sparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };

        let twiddles = unsafe { SliceAccessor::new(self.twiddles.as_slice()) };

        let neg_mask2: f32x8 = unsafe {
            mem::transmute(u32x8::new(
                0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0,
            ))
        };

        let pre_twiddle = sparams.kernel_type() == KernelType::Dit;
        let post_twiddle = sparams.kernel_type() == KernelType::Dif;

        for x in range_step(0, cparams.size * 2, cparams.unit * 8) {
            for y in 0..cparams.unit / 4 {
                let cur1 = &mut data[x + y * 8] as *mut f32 as *mut f32x8;
                let cur2 = &mut data[x + y * 8 + cparams.unit * 2] as *mut f32 as *mut f32x8;
                let cur3 = &mut data[x + y * 8 + cparams.unit * 4] as *mut f32 as *mut f32x8;
                let cur4 = &mut data[x + y * 8 + cparams.unit * 6] as *mut f32 as *mut f32x8;

                // riri format
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
                    avx_f32x8_complex_mul_riri(y1, twiddle_1)
                } else {
                    y1
                };
                let z2 = if pre_twiddle {
                    avx_f32x8_complex_mul_riri(z1, twiddle_2)
                } else {
                    z1
                };
                let w2 = if pre_twiddle {
                    avx_f32x8_complex_mul_riri(w1, twiddle_3)
                } else {
                    w1
                };

                // perform size-4 FFT
                let x3 = x2 + z2;
                let y3 = y2 + w2;
                let z3 = x2 - z2;
                let w3t = y2 - w2;

                // w3 = w3t * i
                let w3 =
                    avx_f32x8_bitxor(shuffle!(w3t, w3t, [1, 0, 11, 10, 5, 4, 15, 14]), neg_mask2);

                let (x4, y4, z4, w4) = if sparams.inverse() {
                    (x3 + y3, z3 + w3, x3 - y3, z3 - w3)
                } else {
                    (x3 + y3, z3 - w3, x3 - y3, z3 + w3)
                };

                // apply twiddle factor
                let x5 = x4;
                let y5 = if post_twiddle {
                    avx_f32x8_complex_mul_riri(y4, twiddle_1)
                } else {
                    y4
                };
                let z5 = if post_twiddle {
                    avx_f32x8_complex_mul_riri(z4, twiddle_2)
                } else {
                    z4
                };
                let w5 = if post_twiddle {
                    avx_f32x8_complex_mul_riri(w4, twiddle_3)
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
        32
    }
}

/// This Radix-4 kernel computes eight small FFTs in a single iteration.
#[derive(Debug)]
struct AvxRadix4Kernel4<T: StaticParams> {
    cparams: KernelCreationParams,
    twiddles: Vec<f32x8>,
    sparams: T,
}

impl<T: StaticParams> AvxRadix4Kernel4<T> {
    fn new(cparams: &KernelCreationParams, sparams: T) -> Self {
        sparams.check_param(cparams);
        assert_eq!(cparams.radix, 4);
        assert_eq!(cparams.unit % 8, 0);

        let full_circle = if cparams.inverse { 2f32 } else { -2f32 };
        let mut twiddles = Vec::new();
        for i in range_step(0, cparams.unit, 8) {
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
            let c5 = Complex::new(
                0f32,
                full_circle * (i + 4) as f32 / (cparams.radix * cparams.unit) as f32
                    * f32::consts::PI,
            )
            .exp();
            let c6 = Complex::new(
                0f32,
                full_circle * (i + 5) as f32 / (cparams.radix * cparams.unit) as f32
                    * f32::consts::PI,
            )
            .exp();
            let c7 = Complex::new(
                0f32,
                full_circle * (i + 6) as f32 / (cparams.radix * cparams.unit) as f32
                    * f32::consts::PI,
            )
            .exp();
            let c8 = Complex::new(
                0f32,
                full_circle * (i + 7) as f32 / (cparams.radix * cparams.unit) as f32
                    * f32::consts::PI,
            )
            .exp();
            // rrrrrrrr-iiiiiiiii format
            // 12563478
            twiddles.push(f32x8::new(
                c1.re, c2.re, c5.re, c6.re, c3.re, c4.re, c7.re, c8.re,
            ));
            twiddles.push(f32x8::new(
                c1.im, c2.im, c5.im, c6.im, c3.im, c4.im, c7.im, c8.im,
            ));

            let c12 = c1 * c1;
            let c22 = c2 * c2;
            let c32 = c3 * c3;
            let c42 = c4 * c4;
            let c52 = c5 * c5;
            let c62 = c6 * c6;
            let c72 = c7 * c7;
            let c82 = c8 * c8;
            twiddles.push(f32x8::new(
                c12.re, c22.re, c52.re, c62.re, c32.re, c42.re, c72.re, c82.re,
            ));
            twiddles.push(f32x8::new(
                c12.im, c22.im, c52.im, c62.im, c32.im, c42.im, c72.im, c82.im,
            ));

            let c13 = c12 * c1;
            let c23 = c22 * c2;
            let c33 = c32 * c3;
            let c43 = c42 * c4;
            let c53 = c52 * c5;
            let c63 = c62 * c6;
            let c73 = c72 * c7;
            let c83 = c82 * c8;
            twiddles.push(f32x8::new(
                c13.re, c23.re, c53.re, c63.re, c33.re, c43.re, c73.re, c83.re,
            ));
            twiddles.push(f32x8::new(
                c13.im, c23.im, c53.im, c63.im, c33.im, c43.im, c73.im, c83.im,
            ));
        }

        Self {
            cparams: *cparams,
            twiddles: twiddles,
            sparams: sparams,
        }
    }
}

impl<T: StaticParams> AlignReqKernel<f32> for AvxRadix4Kernel4<T> {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<f32>) {
        let cparams = &self.cparams;
        let sparams = &self.sparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };

        let twiddles = unsafe { SliceAccessor::new(self.twiddles.as_slice()) };
        let pre_twiddle = sparams.kernel_type() == KernelType::Dit;
        let post_twiddle = sparams.kernel_type() == KernelType::Dif;

        for x in range_step(0, cparams.size * 2, cparams.unit * 8) {
            for y in 0..cparams.unit / 8 {
                let cur1a = &mut data[x + y * 16] as *mut f32 as *mut f32x8;
                let cur1b = &mut data[x + y * 16 + 8] as *mut f32 as *mut f32x8;
                let cur2a = &mut data[x + y * 16 + cparams.unit * 2] as *mut f32 as *mut f32x8;
                let cur2b = &mut data[x + y * 16 + cparams.unit * 2 + 8] as *mut f32 as *mut f32x8;
                let cur3a = &mut data[x + y * 16 + cparams.unit * 4] as *mut f32 as *mut f32x8;
                let cur3b = &mut data[x + y * 16 + cparams.unit * 4 + 8] as *mut f32 as *mut f32x8;
                let cur4a = &mut data[x + y * 16 + cparams.unit * 6] as *mut f32 as *mut f32x8;
                let cur4b = &mut data[x + y * 16 + cparams.unit * 6 + 8] as *mut f32 as *mut f32x8;
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

                // convert riririri-riririri to rrrrrrrr-iiiiiiii (vshufps)
                //         1 2 3 4  5 6 7 8     12563478
                let x2r = shuffle!(x1a, x1b, [0, 2, 8, 10, 4, 6, 12, 14]);
                let x2i = shuffle!(x1a, x1b, [1, 3, 9, 11, 5, 7, 13, 15]);
                let y2r = shuffle!(y1a, y1b, [0, 2, 8, 10, 4, 6, 12, 14]);
                let y2i = shuffle!(y1a, y1b, [1, 3, 9, 11, 5, 7, 13, 15]);
                let z2r = shuffle!(z1a, z1b, [0, 2, 8, 10, 4, 6, 12, 14]);
                let z2i = shuffle!(z1a, z1b, [1, 3, 9, 11, 5, 7, 13, 15]);
                let w2r = shuffle!(w1a, w1b, [0, 2, 8, 10, 4, 6, 12, 14]);
                let w2i = shuffle!(w1a, w1b, [1, 3, 9, 11, 5, 7, 13, 15]);

                // apply twiddle factor
                let x3r = x2r;
                let x3i = x2i;
                let y3r = if pre_twiddle {
                    avx_fma_f32x8_fmsub(y2r, twiddle1_r, y2i * twiddle1_i)
                } else {
                    y2r
                };
                let y3i = if pre_twiddle {
                    avx_fma_f32x8_fmadd(y2r, twiddle1_i, y2i * twiddle1_r)
                } else {
                    y2i
                };
                let z3r = if pre_twiddle {
                    avx_fma_f32x8_fmsub(z2r, twiddle2_r, z2i * twiddle2_i)
                } else {
                    z2r
                };
                let z3i = if pre_twiddle {
                    avx_fma_f32x8_fmadd(z2r, twiddle2_i, z2i * twiddle2_r)
                } else {
                    z2i
                };
                let w3r = if pre_twiddle {
                    avx_fma_f32x8_fmsub(w2r, twiddle3_r, w2i * twiddle3_i)
                } else {
                    w2r
                };
                let w3i = if pre_twiddle {
                    avx_fma_f32x8_fmadd(w2r, twiddle3_i, w2i * twiddle3_r)
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
                    avx_fma_f32x8_fmsub(y5r, twiddle1_r, y5i * twiddle1_i)
                } else {
                    y5r
                };
                let y6i = if post_twiddle {
                    avx_fma_f32x8_fmadd(y5r, twiddle1_i, y5i * twiddle1_r)
                } else {
                    y5i
                };
                let z6r = if post_twiddle {
                    avx_fma_f32x8_fmsub(z5r, twiddle2_r, z5i * twiddle2_i)
                } else {
                    z5r
                };
                let z6i = if post_twiddle {
                    avx_fma_f32x8_fmadd(z5r, twiddle2_i, z5i * twiddle2_r)
                } else {
                    z5i
                };
                let w6r = if post_twiddle {
                    avx_fma_f32x8_fmsub(w5r, twiddle3_r, w5i * twiddle3_i)
                } else {
                    w5r
                };
                let w6i = if post_twiddle {
                    avx_fma_f32x8_fmadd(w5r, twiddle3_i, w5i * twiddle3_r)
                } else {
                    w5i
                };

                // convert to rrrrrrrr-iiiiiiii to riririri-riririri (vunpcklps/vunpckups)
                let x7a = shuffle!(x6r, x6i, [0, 8, 1, 9, 4, 12, 5, 13]);
                let x7b = shuffle!(x6r, x6i, [2, 10, 3, 11, 6, 14, 7, 15]);
                let y7a = shuffle!(y6r, y6i, [0, 8, 1, 9, 4, 12, 5, 13]);
                let y7b = shuffle!(y6r, y6i, [2, 10, 3, 11, 6, 14, 7, 15]);
                let z7a = shuffle!(z6r, z6i, [0, 8, 1, 9, 4, 12, 5, 13]);
                let z7b = shuffle!(z6r, z6i, [2, 10, 3, 11, 6, 14, 7, 15]);
                let w7a = shuffle!(w6r, w6i, [0, 8, 1, 9, 4, 12, 5, 13]);
                let w7b = shuffle!(w6r, w6i, [2, 10, 3, 11, 6, 14, 7, 15]);

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
        32
    }
}
