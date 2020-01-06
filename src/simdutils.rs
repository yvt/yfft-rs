
#[cfg(target_arch = "x86")]
use std::arch::x86 as vendor;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as vendor;
use std::mem;
pub use packed_simd::{f32x8, i32x8, u32x8, u64x4, f32x4, i32x4, u32x4};

#[cfg(test)]
use num_complex::Complex;

#[allow(dead_code)]
#[inline]
pub fn f32x4_bitxor(lhs: f32x4, rhs: f32x4) -> f32x4 {
    let x2: i32x4 = unsafe { mem::transmute(lhs) };
    let y2: i32x4 = unsafe { mem::transmute(rhs) };
    let z = x2 ^ y2;
    unsafe { mem::transmute(z) }
}

#[allow(dead_code)]
pub fn f32x4_to_array(x: f32x4) -> [f32; 4] {
    [x.extract(0), x.extract(1), x.extract(2), x.extract(3)]
}

/// `neg_mask` must be `[0x80000000, 0x80000000, 0, 0]`
#[inline]
pub fn f32x4_complex_mul_rrii(x: f32x4, y: f32x4, neg_mask: f32x4) -> f32x4 {
    let y_iirr = shuffle!(y, y, [2, 3, 4, 5]);

    // (y1a.r * ta.r, y1b.r * tb.r, y1a.i * ta.i, y1b.i * tb.i)
    let t2 = x * y;

    // (y1a.r * ta.i, y1b.r * tb.i, y1a.i * ta.r, y1b.i * tb.r)
    let t3 = x * y_iirr;

    // (y1a.r * ta.r, y1b.r * tb.r, y1a.r * ta.i, y1b.r * tb.i)
    let t4 = shuffle!(t2, t3, [0, 1, 4, 5]);

    // (y1a.i * ta.i, y1b.i * tb.i, y1a.i * ta.r, y1b.i * tb.r)
    let t5 = shuffle!(t2, t3, [2, 3, 6, 7]);

    // (-y1a.i * ta.i, -y1b.i * tb.i, y1a.i * ta.r, y1b.i * tb.r)
    let t6 = f32x4_bitxor(t5, neg_mask);

    // (y3a.r, y3b.r, y3a.i, y3b.i) =
    // (y1a.r * ta.r - y1a.i * ta.i, y1b.r * tb.r - y1b.i * tb.i,
    //  y1a.r * ta.i + y1a.i * ta.r, y1b.r * tb.i + y1b.i * tb.r)
    t4 + t6
}

#[test]
fn test_f32x4_complex_mul_rrii() {
    let neg_mask = unsafe { mem::transmute(u32x4::new(0x80000000, 0x80000000, 0, 0)) };

    let c1: Complex<f32> = Complex::new(123f32, 456f32);
    let c2: Complex<f32> = Complex::new(789f32, 135f32);
    let c3: Complex<f32> = Complex::new(114f32, 514f32);
    let c4: Complex<f32> = Complex::new(987f32, 654f32);

    let d1 = c1 * c3;
    let d2 = c2 * c4;

    let x = f32x4::new(c1.re, c2.re, c1.im, c2.im);
    let y = f32x4::new(c3.re, c4.re, c3.im, c4.im);
    let z = f32x4_complex_mul_rrii(x, y, neg_mask);

    assert_eq!(f32x4_to_array(z), [d1.re, d2.re, d1.im, d2.im]);
}

#[cfg(all(target_feature = "sse3", not(target_feature = "fma")))]
#[allow(dead_code)]
pub fn sse3_fma_f32x4_fmaddsub(x: f32x4, y: f32x4, z: f32x4) -> f32x4 {
    let xy = unsafe { mem::transmute(x * y) };
    let z = unsafe { mem::transmute(z) };
    let w = unsafe { vendor::_mm_addsub_ps(xy, z) };
    unsafe { mem::transmute(w) }
}

#[cfg(all(target_feature = "sse3", target_feature = "fma"))]
#[allow(dead_code)]
pub fn sse3_fma_f32x4_fmaddsub(x: f32x4, y: f32x4, z: f32x4) -> f32x4 {
    let x = unsafe { mem::transmute(x) };
    let y = unsafe { mem::transmute(y) };
    let z = unsafe { mem::transmute(z) };
    let w = unsafe { vendor::_mm_fmaddsub_ps(x, y, z) };
    unsafe { mem::transmute(w) }
}

#[cfg(target_feature = "sse3")]
#[inline]
#[allow(dead_code)]
pub fn sse3_f32x4_complex_mul_riri(x: f32x4, y: f32x4) -> f32x4 {
    // (r1, i1, ...) * (r3, i3, ...)
    //   --> ((r1 * r3) - (i1 * i3), (r1 * i3) + (i1 * r3), ...)
    let x1: f32x4 = shuffle!(x, x, [0, 0, 2, 2]); // movsldup
    let x2: f32x4 = shuffle!(x, x, [1, 1, 3, 3]); // movshdup
    let y1: f32x4 = y;
    let y2: f32x4 = shuffle!(y, y, [1, 0, 3, 2]); // shufps
    let z = sse3_fma_f32x4_fmaddsub(x1, y1, x2 * y2); // vaddsubps/vfmaddsubXXXps
    return z;
}

#[cfg(target_feature = "sse3")]
#[test]
#[allow(dead_code)]
fn test_sse3_f32x4_complex_mul_riri() {
    let c1: Complex<f32> = Complex::new(123f32, 456f32);
    let c2: Complex<f32> = Complex::new(789f32, 135f32);
    let c3: Complex<f32> = Complex::new(114f32, 514f32);
    let c4: Complex<f32> = Complex::new(987f32, 654f32);

    let d1 = c1 * c3;
    let d2 = c2 * c4;

    let x = f32x4::new(c1.re, c1.im, c2.re, c2.im);
    let y = f32x4::new(c3.re, c3.im, c4.re, c4.im);
    let z = sse3_f32x4_complex_mul_riri(x, y);

    assert_eq!(f32x4_to_array(z), [d1.re, d1.im, d2.re, d2.im]);
}

#[cfg(all(target_feature = "avx", not(target_feature = "fma")))]
#[allow(dead_code)]
pub fn avx_fma_f32x8_fmadd(x: f32x8, y: f32x8, z: f32x8) -> f32x8 {
    x * y + z
}

#[cfg(all(target_feature = "avx", target_feature = "fma"))]
#[allow(dead_code)]
pub fn avx_fma_f32x8_fmadd(x: f32x8, y: f32x8, z: f32x8) -> f32x8 {
    let x: vendor::__m256 = unsafe { mem::transmute(x) };
    let y: vendor::__m256 = unsafe { mem::transmute(y) };
    let z: vendor::__m256 = unsafe { mem::transmute(z) };
    unsafe { mem::transmute(vendor::_mm256_fmadd_ps(x, y, z)) }
}

#[cfg(all(target_feature = "avx", not(target_feature = "fma")))]
#[allow(dead_code)]
pub fn avx_fma_f32x8_fmsub(x: f32x8, y: f32x8, z: f32x8) -> f32x8 {
    x * y - z
}

#[cfg(all(target_feature = "avx", target_feature = "fma"))]
#[allow(dead_code)]
pub fn avx_fma_f32x8_fmsub(x: f32x8, y: f32x8, z: f32x8) -> f32x8 {
    let x: vendor::__m256 = unsafe { mem::transmute(x) };
    let y: vendor::__m256 = unsafe { mem::transmute(y) };
    let z: vendor::__m256 = unsafe { mem::transmute(z) };
    unsafe { mem::transmute(vendor::_mm256_fmsub_ps(x, y, z)) }
}

#[cfg(all(target_feature = "avx", not(target_feature = "fma")))]
#[allow(dead_code)]
pub fn avx_fma_f32x8_fmaddsub(x: f32x8, y: f32x8, z: f32x8) -> f32x8 {
    let xy: vendor::__m256 = unsafe { mem::transmute(x * y) };
    let z: vendor::__m256 = unsafe { mem::transmute(z) };
    unsafe { mem::transmute(vendor::_mm256_addsub_ps(xy, z)) }
}

#[cfg(all(target_feature = "avx", target_feature = "fma"))]
#[allow(dead_code)]
pub fn avx_fma_f32x8_fmaddsub(x: f32x8, y: f32x8, z: f32x8) -> f32x8 {
    let x: vendor::__m256 = unsafe { mem::transmute(x) };
    let y: vendor::__m256 = unsafe { mem::transmute(y) };
    let z: vendor::__m256 = unsafe { mem::transmute(z) };
    unsafe { mem::transmute(vendor::_mm256_fmaddsub_ps(x, y, z)) }
}

#[cfg(target_feature = "avx")]
#[inline]
#[allow(dead_code)]
pub fn avx_f32x8_complex_mul_riri(x: f32x8, y: f32x8) -> f32x8 {
    // (r1, i1, ...) * (r3, i3, ...)
    //   --> ((r1 * r3) - (i1 * i3), (r1 * i3) + (i1 * r3), ...)
    let x1: f32x8 = shuffle!(x, x, [0, 0, 2, 2, 4, 4, 6, 6]); // vmovsldup
    let x2: f32x8 = shuffle!(x, x, [1, 1, 3, 3, 5, 5, 7, 7]); // vmovshdup
    let y1: f32x8 = y;
    let y2: f32x8 = shuffle!(y, y, [1, 0, 3, 2, 5, 4, 7, 6]); // vpermilps
    let z = avx_fma_f32x8_fmaddsub(x1, y1, x2 * y2); // vaddsubps/vfmaddsubXXXps
    return z;
}

#[cfg(target_feature = "avx")]
#[test]
#[allow(dead_code)]
fn test_avx_f32x8_complex_mul_riri() {
    let c1: Complex<f32> = Complex::new(123f32, 456f32);
    let c2: Complex<f32> = Complex::new(789f32, 135f32);
    let c3: Complex<f32> = Complex::new(114f32, 514f32);
    let c4: Complex<f32> = Complex::new(987f32, 654f32);
    let c5: Complex<f32> = Complex::new(12f32, 46f32);
    let c6: Complex<f32> = Complex::new(78f32, 15f32);
    let c7: Complex<f32> = Complex::new(11f32, 54f32);
    let c8: Complex<f32> = Complex::new(98f32, 64f32);

    let d1 = c1 * c3;
    let d2 = c2 * c4;
    let d3 = c5 * c7;
    let d4 = c6 * c8;

    let x = f32x8::new(c1.re, c1.im, c2.re, c2.im, c5.re, c5.im, c6.re, c6.im);
    let y = f32x8::new(c3.re, c3.im, c4.re, c4.im, c7.re, c7.im, c8.re, c8.im);
    let z = avx_f32x8_complex_mul_riri(x, y);

    assert_eq!(
        f32x8_to_array(z),
        [d1.re, d1.im, d2.re, d2.im, d3.re, d3.im, d4.re, d4.im]
    );
}

#[cfg(target_feature = "avx")]
#[allow(dead_code)]
#[inline]
pub fn avx_f32x8_bitxor(lhs: f32x8, rhs: f32x8) -> f32x8 {
    let x2: i32x8 = unsafe { mem::transmute(lhs) };
    let y2: i32x8 = unsafe { mem::transmute(rhs) };
    let z = x2 ^ y2;
    unsafe { mem::transmute(z) }
}

#[cfg(target_feature = "avx")]
#[allow(dead_code)]
pub fn f32x8_to_array(x: f32x8) -> [f32; 8] {
    [
        x.extract(0),
        x.extract(1),
        x.extract(2),
        x.extract(3),
        x.extract(4),
        x.extract(5),
        x.extract(6),
        x.extract(7),
    ]
}
