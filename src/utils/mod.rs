use itertools::Itertools;
use rand::{distributions::Standard, prelude::Distribution, RngCore};

pub mod arithmetic;
pub mod bitreverse;
pub mod transpose;

pub use arithmetic::*;
pub use bitreverse::*;

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

#[inline(always)]
pub fn log2_strict(n: usize) -> usize {
    let res = n.trailing_zeros();
    debug_assert_eq!(n.wrapping_shr(res), 1, "Not a power of two: {n}");
    res as usize
}

pub trait TwoAdicSlice<T>: core::ops::Deref<Target = [T]> {
    #[inline(always)]
    fn k(&self) -> usize {
        log2_strict(self.len())
    }
}

impl<V> TwoAdicSlice<V> for Vec<V> {}
impl<V> TwoAdicSlice<V> for &[V] {}
impl<V> TwoAdicSlice<V> for &mut [V] {}

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
        if base_chunk_size != 0 {
            for (chunk_id, chunk) in v_lo.chunks_exact_mut(base_chunk_size).enumerate() {
                let offset = split_pos + (chunk_id * base_chunk_size);
                scope.spawn(move |_| f(chunk, offset));
            }
        }
    });
}
