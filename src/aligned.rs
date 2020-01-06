use std::{
    fmt,
    mem::size_of,
    ops::{Deref, DerefMut},
};

/// The alignment value guaranteed by `AlignedVec`.
const ALIGN: usize = 32;

fn ptr_lsbs(x: usize) -> usize {
    x & (ALIGN - 1)
}

/// Provides a subset of `Vec`'s interface while providing a minimum alignment
/// guarantee that is convenient for SIMD operations.
pub struct AlignedVec<T> {
    storage: Vec<T>,
    offset: usize,
}

impl<T: Copy + Default> AlignedVec<T> {
    pub fn with_capacity(i: usize) -> Self {
        debug_assert!(size_of::<T>() <= ALIGN);
        debug_assert!(ALIGN % size_of::<T>() == 0);

        let mut storage: Vec<T> = Vec::with_capacity(i + ALIGN / size_of::<T>() - 1);
        let mut offset = 0;

        // Increase the padding until the storage is aligned
        while ptr_lsbs(storage.as_ptr().wrapping_add(offset) as _) != 0 {
            storage.push(T::default());
            offset += 1;

            debug_assert!(offset < ALIGN / size_of::<T>());
        }

        Self { storage, offset }
    }

    pub fn push(&mut self, x: T) {
        if self.storage.len() >= self.storage.capacity() {
            panic!("collection is full");
        }
        self.storage.push(x);
    }
}

impl<T: fmt::Debug> fmt::Debug for AlignedVec<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("AlignedVec")
            .field("offset", &self.offset)
            .field("entries", &&self[..])
            .finish()
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.storage[self.offset..]
    }
}

impl<T> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.storage[self.offset..]
    }
}
