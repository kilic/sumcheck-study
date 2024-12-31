use rand::{distributions::Standard, prelude::Distribution};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    field::{ExtField, Field},
    utils::{log2_strict, n_rand, TwoAdicSlice},
    BitReversed, IndexOrder, Natural,
};

impl<V: Field, Index: IndexOrder> core::ops::Deref for MLE<V, Index> {
    type Target = Vec<V>;

    fn deref(&self) -> &Self::Target {
        &self.evals
    }
}

impl<V: Field, Index: IndexOrder> core::ops::DerefMut for MLE<V, Index> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.evals
    }
}

fn eq<F: Field, I: IndexOrder>(points: &[F]) -> MLE<F, I> {
    let k = points.len();
    let mut ml = MLE::zero(k);
    ml[0] = F::ONE;
    for (i, &point) in points.iter().enumerate() {
        let mid = 1 << i;
        let (lo, hi) = ml.split_at_mut(mid);
        lo.par_iter_mut()
            .zip(hi.par_iter_mut())
            .for_each(|(lo, hi)| {
                let u = *lo * point;
                *hi = u;
                *lo -= u;
            });
    }
    ml
}

fn fix_mut<F: Field, I: IndexOrder>(mle: &mut MLE<F, I>, points: &[F]) {
    assert!(!points.is_empty());
    let k = mle.k();
    points.iter().enumerate().for_each(|(i, r)| {
        for dst in 0..(1 << (k - i - 1)) {
            let src = dst << 1;
            let a0 = mle[src];
            let a1 = mle[src + 1];
            mle[dst] = *r * (a1 - a0) + a0;
        }
    });
    mle.drop(1 << (mle.k() - points.len()));
}

fn fix_backwards_mut<F: Field, I: IndexOrder>(mle: &mut MLE<F, I>, points: &[F]) {
    assert!(!points.is_empty());
    points.iter().for_each(|r| {
        let (lo, hi) = mle.split_mut_half();
        lo.par_iter_mut()
            .zip_eq(hi.par_iter())
            .for_each(|(lo, hi)| *lo += *r * (*hi - *lo));
        mle.drop_hi();
    });
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MLE<V, Index: IndexOrder> {
    evals: Vec<V>,
    order: Index,
}

impl<F: Field, Index: IndexOrder> IntoIterator for MLE<F, Index> {
    type Item = F;
    type IntoIter = std::vec::IntoIter<F>;

    fn into_iter(self) -> Self::IntoIter {
        self.evals.into_iter()
    }
}

impl<F: Field, Index: IndexOrder> From<&[F]> for MLE<F, Index> {
    fn from(evals: &[F]) -> Self {
        Self::from_slice(evals)
    }
}

impl<F: Field, Index: IndexOrder> From<Vec<F>> for MLE<F, Index> {
    fn from(evals: Vec<F>) -> Self {
        Self::new(evals)
    }
}

impl<F: Field> MLE<F, BitReversed> {
    pub fn extend(&mut self, point: F) {
        let k = self.k();
        self.resize(1 << (k + 1), F::ZERO);
        let (lo, hi) = self.split_at_mut(1 << k);
        lo.par_iter_mut()
            .zip_eq(hi.par_iter_mut())
            .for_each(|(lo, hi)| {
                *hi = *lo * point;
                *lo -= *hi;
            });
    }

    pub fn eq(points: &[F]) -> Self {
        let mut points = points.to_vec();
        points.reverse();
        eq(&points)
    }

    pub fn fix_mut(&mut self, points: &[F]) {
        fix_backwards_mut(self, points);
    }

    pub fn fix_backwards_mut(&mut self, points: &[F]) {
        fix_mut(self, points);
    }

    pub fn eval<EF: ExtField<F>>(&self, points: &[EF]) -> EF {
        assert_eq!(points.len(), self.k());
        let eq = MLE::<_, BitReversed>::eq(points);
        eq.par_iter()
            .zip(self.par_iter())
            .map(|(&a, &b)| a * b)
            .sum()
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn bit_reverse(&mut self) -> MLE<F, Natural> {
        use crate::utils::BitReverse;

        self.evals.reverse_bits();
        MLE {
            evals: self.evals.clone(),
            order: Natural,
        }
    }
}

impl<F: Field> MLE<F, Natural> {
    pub fn extend(&mut self, point: F) {
        let k = self.k();
        self.resize(1 << (k + 1), F::ZERO);
        let mid = 1 << k;
        for j in 0..mid {
            let u = self[j] * point;
            self[j + mid] = u;
            self[j] -= u;
        }
    }

    pub fn fix_mut(&mut self, points: &[F]) {
        fix_mut(self, points);
    }

    pub fn fix_backwards_mut(&mut self, points: &[F]) {
        fix_backwards_mut(self, points);
    }

    pub fn eq(points: &[F]) -> Self {
        eq(points)
    }

    pub fn eval<EF: ExtField<F>>(&self, points: &[EF]) -> EF {
        assert_eq!(points.len(), self.k());
        let eq = MLE::<_, Natural>::eq(points);

        eq.par_iter()
            .zip(self.par_iter())
            .map(|(&a, &b)| a * b)
            .sum()
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn bit_reverse(&mut self) -> MLE<F, BitReversed> {
        use crate::utils::BitReverse;

        self.evals.reverse_bits();
        MLE {
            evals: self.evals.clone(),
            order: BitReversed,
        }
    }
}

impl<F: Field, Index: IndexOrder> MLE<F, Index> {
    pub fn new(evals: Vec<F>) -> Self {
        let _ = evals.k();
        Self {
            evals,
            order: Index::default(),
        }
    }

    pub fn zero(k: usize) -> Self {
        Self::new(vec![F::ZERO; 1 << k])
    }

    pub fn from_slice(evals: &[F]) -> Self {
        Self::new(evals.to_vec())
    }

    pub fn k(&self) -> usize {
        log2_strict(self.len())
    }

    pub fn rand(rng: impl rand::RngCore, k: usize) -> Self
    where
        Standard: Distribution<F>,
    {
        n_rand(rng, 1 << k).into()
    }

    pub fn inner_prod<EF: ExtField<F>>(&self, rhs: &MLE<EF, Index>) -> EF {
        rhs.par_iter()
            .zip(self.par_iter())
            .map(|(&a, &b)| a * b)
            .sum()
    }

    pub fn split_mut_half(&mut self) -> (&mut [F], &mut [F]) {
        let k = self.k() - 1;
        self.split_at_mut(1 << k)
    }

    pub fn drop_hi(&mut self) {
        let k = self.k() - 1;
        self.drop(k);
    }

    pub fn drop(&mut self, k: usize) {
        assert!(k <= self.k());
        self.truncate(1 << k);
    }
}

#[cfg(test)]
mod test {
    use crate::{
        mle::MLE,
        test::seed_rng,
        utils::{n_rand, BitReverse},
        BitReversed, Natural,
    };

    type F = crate::field::goldilocks::Goldilocks;

    #[test]
    fn test_mle() {
        let mut rng = seed_rng();
        let k = 10;
        let evals = n_rand::<F>(&mut rng, 1 << k);
        let r = n_rand::<F>(&mut rng, k);

        {
            let eq0 = MLE::<F, Natural>::eq(&r);
            let mut eq1 = MLE::<F, BitReversed>::eq(&r);
            let eq1 = eq1.bit_reverse();
            assert_eq!(eq0, eq1);
        }

        let e0 = {
            let mut a0 = MLE::<F, Natural>::new(evals.clone());
            let mut a1 = a0.clone();
            let e0 = a0.eval(&r[..]);
            a0.fix_mut(&r[..]);
            assert_eq!(e0, a0.first().copied().unwrap());

            let mut r = r.clone();
            r.reverse();
            a1.fix_backwards_mut(&r[..]);
            assert_eq!(e0, a1.first().copied().unwrap());

            e0
        };

        {
            let mut evals = evals.clone();
            evals.reverse_bits();
            let mut a0 = MLE::<F, BitReversed>::new(evals);
            let mut a1 = a0.clone();
            let e1 = a0.eval(&r[..]);
            assert_eq!(e0, e1);
            a0.fix_mut(&r[..]);
            assert_eq!(e0, a0.first().copied().unwrap());

            let mut r = r.clone();
            r.reverse();
            a1.fix_backwards_mut(&r[..]);
            assert_eq!(e0, a1.first().copied().unwrap());
        }
    }
}
