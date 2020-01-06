use super::super::Num;
use super::utils::{if_compatible, AlignInfo, AlignReqKernel, AlignReqKernelWrapper};
use super::{Kernel, KernelParams, SliceAccessor};

use packed_simd::{u32x4, u64x4, u64x2};

use std::{mem, ptr};

pub unsafe fn new_x86_avx_bit_reversal_kernel<T>(indices: &Vec<usize>) -> Option<Box<Kernel<T>>>
where
    T: Num,
{
    if indices.len() < 8 {
        // doesn't benefit much
        return None;
    }

    if_compatible(|| {
        Some(
            Box::new(AlignReqKernelWrapper::new(AvxDWordBitReversalKernel {
                indices: indices.clone(),
            })) as Box<Kernel<f32>>,
        )
    })
}

#[derive(Debug)]
struct AvxDWordBitReversalKernel {
    indices: Vec<usize>,
}

impl<T: Num> AlignReqKernel<T> for AvxDWordBitReversalKernel {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<T>) {
        assert_eq!(mem::size_of::<T>(), 4);

        let indices = unsafe { SliceAccessor::new(&self.indices) };
        let size = self.indices.len();
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..size * 2]) };
        let mut wa = unsafe { SliceAccessor::new(&mut params.work_area[0..size * 2]) };
        wa.copy_from_slice(*data);
        let mut i = 0;
        while i + 7 < size {
            let index1 = indices[i];
            let index2 = indices[i + 1];
            let index3 = indices[i + 2];
            let index4 = indices[i + 3];
            let index5 = indices[i + 4];
            let index6 = indices[i + 5];
            let index7 = indices[i + 6];
            let index8 = indices[i + 7];

            let src1: *const u64 = &wa[index1 * 2] as *const T as *const u64;
            let src2: *const u64 = &wa[index2 * 2] as *const T as *const u64;
            let src3: *const u64 = &wa[index3 * 2] as *const T as *const u64;
            let src4: *const u64 = &wa[index4 * 2] as *const T as *const u64;
            let src5: *const u64 = &wa[index5 * 2] as *const T as *const u64;
            let src6: *const u64 = &wa[index6 * 2] as *const T as *const u64;
            let src7: *const u64 = &wa[index7 * 2] as *const T as *const u64;
            let src8: *const u64 = &wa[index8 * 2] as *const T as *const u64;
            let dest: *mut u64x4 = &mut data[i * 2] as *mut T as *mut u64x4;
            unsafe {
                I::write(dest, u64x4::new(*src1, *src2, *src3, *src4));
                I::write(dest.offset(1), u64x4::new(*src5, *src6, *src7, *src8));
            }

            i += 8;
        }
        while i < size {
            let index = indices[i];

            let src: *const u64 = &wa[index * 2] as *const T as *const u64;
            let dest: *mut u64 = &mut data[i * 2] as *mut T as *mut u64;
            unsafe {
                *dest = *src;
            }

            i += 1;
        }
    }
    fn required_work_area_size(&self) -> usize {
        self.indices.len() * 2
    }
    fn alignment_requirement(&self) -> usize {
        32
    }
}

pub unsafe fn new_x86_avx_radix2_bit_reversal_kernel<T>(
    indices: &Vec<usize>,
) -> Option<Box<Kernel<T>>>
where
    T: Num,
{
    if indices.len() < 8 || indices.len() % 8 != 0 {
        // doesn't benefit much / requires an unaligned access
        return None;
    }

    // Check legibility
    for i in 0..indices.len() / 2 {
        if indices[i] > (<u32>::max_value() / 2) as usize {
            return None;
        }
    }

    let (f1, f2, f3) = (
        indices[1] - indices[0],
        indices[2] - indices[0],
        indices[3] - indices[0],
    );
    for i in 0..indices.len() / 8 {
        let (b0, b1, b2, b3) = (
            indices[i * 4],
            indices[i * 4 + 1],
            indices[i * 4 + 2],
            indices[i * 4 + 3],
        );
        if b1 != b0 + f1 || b2 != b0 + f2 || b3 != b0 + f3 {
            return None;
        }
    }

    if_compatible(|| {
        Some(Box::new(AlignReqKernelWrapper::new(
            AvxDWordRadix2BitReversalKernel {
                indices: (0..indices.len() / 8)
                    .map(|i| (indices[i * 4] as u32) * 2)
                    .collect(),
                offs: u32x4::new(0, f1 as u32 * 2, f2 as u32 * 2, f3 as u32 * 2),
            },
        )) as Box<Kernel<f32>>)
    })
}

#[derive(Debug)]
struct AvxDWordRadix2BitReversalKernel {
    indices: Vec<u32>,
    offs: u32x4,
}

impl<T: Num> AlignReqKernel<T> for AvxDWordRadix2BitReversalKernel {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<T>) {
        assert_eq!(mem::size_of::<T>(), 4);

        let indices = unsafe { SliceAccessor::new(&self.indices) };
        let size = self.indices.len();
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..size * 16]) };
        let mut wa = unsafe { SliceAccessor::new(&mut params.work_area[0..size * 16]) };
        wa.copy_from_slice(*data);

        let offs = self.offs;

        let mut i = 0;
        while i < size {
            let index1234 = offs + u32x4::splat(indices[i]);
            let index1 = index1234.extract(0) as usize;
            let index2 = index1234.extract(1) as usize;
            let index3 = index1234.extract(2) as usize;
            let index4 = index1234.extract(3) as usize;

            let src1 = unsafe { ptr::read_unaligned(&wa[index1] as *const T as *const u64x2) };
            let src2 = unsafe { ptr::read_unaligned(&wa[index2] as *const T as *const u64x2) };
            let src3 = unsafe { ptr::read_unaligned(&wa[index3] as *const T as *const u64x2) };
            let src4 = unsafe { ptr::read_unaligned(&wa[index4] as *const T as *const u64x2) };

            let t1a: u64x2 = shuffle!(src1, src2, [0, 2]); // unpcklpd
            let t2a: u64x2 = shuffle!(src3, src4, [0, 2]); // unpcklpd

            let t1b: u64x2 = shuffle!(src1, src2, [1, 3]); // unpckhpd
            let t2b: u64x2 = shuffle!(src3, src4, [1, 3]); // unpckhpd

            let out1: u64x4 = shuffle!(t1a, t2a, [0, 1, 2, 3]); // inserti128
            let out2: u64x4 = shuffle!(t1b, t2b, [0, 1, 2, 3]); // inserti128

            let dest1: *mut u64x4 = &mut data[i * 8] as *mut T as *mut u64x4;
            let dest2: *mut u64x4 = &mut data[(i + size) * 8] as *mut T as *mut u64x4;

            unsafe {
                I::write(dest1, out1);
                I::write(dest2, out2);
            }

            i += 1;
        }
    }
    fn required_work_area_size(&self) -> usize {
        self.indices.len() * 16
    }
    fn alignment_requirement(&self) -> usize {
        32
    }
}

pub unsafe fn new_x86_avx_radix4_bit_reversal_kernel<T>(
    indices: &Vec<usize>,
) -> Option<Box<Kernel<T>>>
where
    T: Num,
{
    if indices.len() < 32 || indices.len() % 32 != 0 {
        // doesn't benefit much / requires an unaligned access
        return None;
    }

    // Check legibility
    for i in 0..indices.len() / 4 {
        if indices[i] > (<u32>::max_value() / 2) as usize {
            return None;
        }
    }

    let (f1, f2, f3) = (
        indices[1] - indices[0],
        indices[2] - indices[0],
        indices[3] - indices[0],
    );
    for i in 0..indices.len() / 16 {
        let (b0, b1, b2, b3) = (
            indices[i * 4],
            indices[i * 4 + 1],
            indices[i * 4 + 2],
            indices[i * 4 + 3],
        );
        if b1 != b0 + f1 || b2 != b0 + f2 || b3 != b0 + f3 {
            return None;
        }
    }

    if_compatible(|| {
        Some(Box::new(AlignReqKernelWrapper::new(
            AvxDWordRadix4BitReversalKernel {
                indices: (0..indices.len() / 16)
                    .map(|i| (indices[i * 4] as u32) * 2)
                    .collect(),
                offs: u32x4::new(0, f1 as u32 * 2, f2 as u32 * 2, f3 as u32 * 2),
            },
        )) as Box<Kernel<f32>>)
    })
}

#[derive(Debug)]
struct AvxDWordRadix4BitReversalKernel {
    indices: Vec<u32>,
    offs: u32x4,
}

impl<T: Num> AlignReqKernel<T> for AvxDWordRadix4BitReversalKernel {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<T>) {
        assert_eq!(mem::size_of::<T>(), 4);

        let indices = unsafe { SliceAccessor::new(&self.indices) };
        let size = self.indices.len();
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..size * 32]) };
        let mut wa = unsafe { SliceAccessor::new(&mut params.work_area[0..size * 32]) };
        wa.copy_from_slice(*data);

        let offs = self.offs;

        let mut i = 0;
        while i + 1 < size {
            for _ in 0..2 {
                let index1234 = offs + u32x4::splat(indices[i]);
                let index1 = index1234.extract(0) as usize;
                let index2 = index1234.extract(1) as usize;
                let index3 = index1234.extract(2) as usize;
                let index4 = index1234.extract(3) as usize;

                let src1 = unsafe { ptr::read_unaligned(&wa[index1] as *const T as *const u64x4) };
                let src2 = unsafe { ptr::read_unaligned(&wa[index2] as *const T as *const u64x4) };
                let src3 = unsafe { ptr::read_unaligned(&wa[index3] as *const T as *const u64x4) };
                let src4 = unsafe { ptr::read_unaligned(&wa[index4] as *const T as *const u64x4) };

                let t1a: u64x4 = shuffle!(src1, src2, [0, 4, 2, 6]); // unpcklpd
                let t2a: u64x4 = shuffle!(src3, src4, [0, 4, 2, 6]); // unpcklpd

                let t1b: u64x4 = shuffle!(src1, src2, [1, 5, 3, 7]); // unpckhpd
                let t2b: u64x4 = shuffle!(src3, src4, [1, 5, 3, 7]); // unpckhpd

                let out1: u64x4 = shuffle!(t1a, t2a, [0, 1, 4, 5]); // inserti128/perm2f128
                let out2: u64x4 = shuffle!(t1b, t2b, [0, 1, 4, 5]); // inserti128/perm2f128
                let out3: u64x4 = shuffle!(t1a, t2a, [2, 3, 6, 7]); // inserti128/perm2f128
                let out4: u64x4 = shuffle!(t1b, t2b, [2, 3, 6, 7]); // inserti128/perm2f128

                let dest1: *mut u64x4 = &mut data[i * 8] as *mut T as *mut u64x4;
                let dest2: *mut u64x4 = &mut data[(i + size) * 8] as *mut T as *mut u64x4;
                let dest3: *mut u64x4 = &mut data[(i + size * 2) * 8] as *mut T as *mut u64x4;
                let dest4: *mut u64x4 = &mut data[(i + size * 3) * 8] as *mut T as *mut u64x4;

                unsafe {
                    I::write(dest1, out1);
                    I::write(dest2, out2);
                    I::write(dest3, out3);
                    I::write(dest4, out4);
                }

                i += 1;
            }
        }
        assert_eq!(i, size);
    }
    fn required_work_area_size(&self) -> usize {
        self.indices.len() * 32
    }
    fn alignment_requirement(&self) -> usize {
        32
    }
}
