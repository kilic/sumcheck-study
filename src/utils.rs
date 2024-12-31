use itertools::Itertools;
use rand::{distributions::Standard, prelude::Distribution, RngCore};

#[macro_export]
macro_rules! split128 {
    ($x:expr) => {
        ($x as u64, ($x >> 64) as u64)
    };
}

pub type Vec2D<T> = Vec<Vec<T>>;
pub type Vec3D<T> = Vec<Vec<Vec<T>>>;

pub fn n_rand<F>(mut rng: impl RngCore, n: usize) -> Vec<F>
where
    Standard: Distribution<F>,
{
    use rand::Rng;
    std::iter::repeat_with(|| rng.gen()).take(n).collect_vec()
}

#[inline]
pub fn log2_strict(n: usize) -> usize {
    let res = n.trailing_zeros();
    assert_eq!(n.wrapping_shr(res), 1, "Not a power of two: {n}");
    res as usize
}

#[inline]
pub const fn reverse_bits(x: usize, range: usize) -> usize {
    x.reverse_bits()
        .overflowing_shr(usize::BITS - range as u32)
        .0
}

pub trait TwoAdicSlice<T>: core::ops::Deref<Target = [T]> {
    fn k(&self) -> usize {
        log2_strict(self.len())
    }
}

impl<V> TwoAdicSlice<V> for Vec<V> {}
impl<V> TwoAdicSlice<V> for &[V] {}
impl<V> TwoAdicSlice<V> for &mut [V] {}

pub trait BitReverse<V>: TwoAdicSlice<V> {
    fn reverse_bits(self);
    fn reverse_bits_2d(self, w: usize);
    fn swap_rows(self, i: usize, j: usize, w: usize);
}

impl<V> BitReverse<V> for &mut [V] {
    fn reverse_bits(self) {
        let n = self.len();
        let k = log2_strict(n);
        (0..n).for_each(|i| {
            let j = reverse_bits(i, k);
            (i < j).then(|| self.swap(i, j));
        });
    }

    fn reverse_bits_2d(self, w: usize) {
        let n = self.len() / w;
        let k = log2_strict(n);

        (0..n).for_each(|i| {
            let j = reverse_bits(i, k);
            (i < j).then(|| self.swap_rows(i, j, w));
        });
    }

    fn swap_rows(self, i: usize, j: usize, w: usize) {
        let (upper, lower) = self.split_at_mut(j * w);
        upper[i * w..(i + 1) * w].swap_with_slice(&mut lower[..w]);
    }
}

#[macro_export]
macro_rules! impl_sub {
    ($lhs:ident) => {
        impl<'a> core::ops::Sub<&'a $lhs> for $lhs {
            type Output = $lhs;

            #[inline]
            fn sub(self, rhs: &'a $lhs) -> $lhs {
                self - *rhs
            }
        }

        impl<'a> core::ops::Sub<$lhs> for &'a $lhs {
            type Output = $lhs;

            #[inline]
            fn sub(self, rhs: $lhs) -> $lhs {
                *self - rhs
            }
        }

        impl<'a, 'b> core::ops::Sub<&'b $lhs> for &'a $lhs {
            type Output = $lhs;

            #[inline]
            fn sub(self, rhs: &'b $lhs) -> $lhs {
                *self - *rhs
            }
        }

        impl core::ops::SubAssign for $lhs {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl core::ops::SubAssign<&$lhs> for $lhs {
            #[inline]
            fn sub_assign(&mut self, rhs: &$lhs) {
                *self -= *rhs;
            }
        }
    };
}

#[macro_export]
macro_rules! impl_add {
    ($lhs:ident) => {
        impl<'a> core::ops::Add<&'a $lhs> for $lhs {
            type Output = $lhs;

            #[inline]
            fn add(self, rhs: &'a $lhs) -> $lhs {
                self + *rhs
            }
        }

        impl<'a> core::ops::Add<$lhs> for &'a $lhs {
            type Output = $lhs;

            #[inline]
            fn add(self, rhs: $lhs) -> $lhs {
                *self + rhs
            }
        }

        impl<'a, 'b> core::ops::Add<&'b $lhs> for &'a $lhs {
            type Output = $lhs;

            #[inline]
            fn add(self, rhs: &'b $lhs) -> $lhs {
                *self + *rhs
            }
        }

        impl core::ops::AddAssign for $lhs {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl core::ops::AddAssign<&$lhs> for $lhs {
            #[inline]
            fn add_assign(&mut self, rhs: &$lhs) {
                *self += *rhs;
            }
        }
    };
}

#[macro_export]
macro_rules! impl_mul {
    ($lhs:ident) => {
        impl<'a> core::ops::Mul<&'a $lhs> for $lhs {
            type Output = $lhs;

            #[inline]
            fn mul(self, rhs: &'a $lhs) -> $lhs {
                self * *rhs
            }
        }

        impl<'a> core::ops::Mul<$lhs> for &'a $lhs {
            type Output = $lhs;

            #[inline]
            fn mul(self, rhs: $lhs) -> $lhs {
                *self * rhs
            }
        }

        impl<'a, 'b> core::ops::Mul<&'b $lhs> for &'a $lhs {
            type Output = $lhs;

            #[inline]
            fn mul(self, rhs: &'b $lhs) -> $lhs {
                *self * *rhs
            }
        }

        impl core::ops::MulAssign for $lhs {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl core::ops::MulAssign<&$lhs> for $lhs {
            #[inline]
            fn mul_assign(&mut self, rhs: &$lhs) {
                *self *= *rhs;
            }
        }
    };
}

pub fn parallelize<T: Send, F: Fn(&mut [T], usize) + Send + Sync + Clone>(v: &mut [T], f: F) {
    let f = &f;
    let total_iters = v.len();
    let num_threads = rayon::current_num_threads();
    let base_chunk_size = total_iters / num_threads;
    let cutoff_chunk_id = total_iters % num_threads;
    let split_pos = cutoff_chunk_id * (base_chunk_size + 1);
    let (v_hi, v_lo) = v.split_at_mut(split_pos);

    rayon::scope(|scope| {
        if cutoff_chunk_id != 0 {
            for (chunk_id, chunk) in v_hi.chunks_exact_mut(base_chunk_size + 1).enumerate() {
                let offset = chunk_id * (base_chunk_size + 1);
                scope.spawn(move |_| f(chunk, offset));
            }
        }
        // Skip special-case: less iterations than number of threads.
        if base_chunk_size != 0 {
            for (chunk_id, chunk) in v_lo.chunks_exact_mut(base_chunk_size).enumerate() {
                let offset = split_pos + (chunk_id * base_chunk_size);
                scope.spawn(move |_| f(chunk, offset));
            }
        }
    });
}
