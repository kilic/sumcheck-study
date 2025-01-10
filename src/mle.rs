use crate::{
    field::{ExtField, Field},
    utils::{log2_strict, n_rand, TwoAdicSlice},
};
use rand::{distributions::Standard, prelude::Distribution};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};

impl<V> core::ops::Deref for MLE<V> {
    type Target = Vec<V>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<V> core::ops::DerefMut for MLE<V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

fn fix_mut<F: Field>(mle: &mut MLE<F>, points: &[F]) {
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

fn fix_backwards_mut<F: Field>(mle: &mut MLE<F>, points: &[F]) {
    assert!(!points.is_empty());
    points.iter().for_each(|r| {
        let (lo, hi) = mle.split_mut_half();
        lo.par_iter_mut()
            .zip_eq(hi.par_iter())
            .for_each(|(lo, hi)| *lo += *r * (*hi - *lo));
        mle.drop_hi();
    });
}

// pub fn eq_xy_eval<F: Field>(x: &[F], y: &[F]) -> F {
//     assert_eq!(x.len(), y.len());
//     x.par_iter()
//         .zip(y.par_iter())
//         .map(|(&xi, &yi)| xi * yi + (F::ONE - xi) * (F::ONE - yi))
//         .product()
// }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MLE<F>(Vec<F>);

impl<F> IntoIterator for MLE<F> {
    type Item = F;
    type IntoIter = std::vec::IntoIter<F>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<F: Clone> From<&[F]> for MLE<F> {
    fn from(evals: &[F]) -> Self {
        Self::from_slice(evals)
    }
}

impl<F> From<Vec<F>> for MLE<F> {
    fn from(evals: Vec<F>) -> Self {
        Self::new(evals)
    }
}

pub fn extend<F: Field>(eq: &mut MLE<F>, point: F) {
    let k = eq.k();
    let (lo, hi) = eq.split_at_mut(1 << k);
    lo.par_iter_mut()
        .zip_eq(hi.par_iter_mut())
        .for_each(|(lo, hi)| {
            *hi = *lo * point;
            *lo -= *hi;
        });
}

impl<F: Field> MLE<F> {
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

    pub fn extend_to(&self, point: F) -> MLE<F> {
        let k = self.k();

        let mut eq = MLE::zero(k + 1);
        eq.par_chunks_mut(2)
            .zip(self.par_iter())
            .for_each(|(eq, &this)| {
                eq[1] = this * point;
                eq[0] = this - eq[1];
            });
        eq
    }

    #[tracing::instrument(level = "info", skip_all, fields(k = points.len()))]
    pub fn eq(points: &[F]) -> Self {
        let k = points.len();
        let mut ml = MLE::zero(k);
        ml[0] = F::ONE;
        for (i, &point) in points.iter().enumerate() {
            let (lo, hi) = ml.split_at_mut(1 << i);
            lo.par_iter_mut()
                .zip(hi.par_iter_mut())
                .for_each(|(lo, hi)| {
                    *hi = *lo * point;
                    *lo -= *hi;
                });
        }
        ml
    }

    pub fn fix_mut(&mut self, points: &[F]) {
        fix_mut(self, points);
    }

    pub fn fix_backwards_mut(&mut self, points: &[F]) {
        fix_backwards_mut(self, points);
    }

    pub fn eval<EF: ExtField<F>>(&self, rs: &[EF]) -> EF {
        assert_eq!(rs.len(), self.k());
        let eq = MLE::<_>::eq(rs);
        eq.par_iter()
            .zip(self.par_iter())
            .map(|(&a, &b)| a * b)
            .sum()
    }

    pub fn reverse_bits(&mut self) {
        use crate::utils::BitReverse;
        self.0.reverse_bits();
    }
}

impl<F> MLE<F> {
    pub fn new(evals: Vec<F>) -> Self {
        let _ = evals.k();
        Self(evals)
    }

    pub fn zero(k: usize) -> Self
    where
        F: Default + Copy,
    {
        Self::new(vec![F::default(); 1 << k])
    }

    pub fn from_slice(evals: &[F]) -> Self
    where
        F: Clone,
    {
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
    use crate::{field::Field, mle::MLE, utils::n_rand};
    type F = crate::field::goldilocks::Goldilocks;

    #[test]
    fn test_mle() {
        let mut rng = crate::test::seed_rng();
        let k = 25;

        let r = n_rand::<F>(&mut rng, k);
        let r_rev = r.iter().copied().rev().collect::<Vec<_>>();

        let eq0 = MLE::<F>::eq(&r);

        let mut eq1 = MLE::<F>::new(vec![F::ONE]);
        r.iter().for_each(|&r| eq1.extend(r));
        assert_eq!(eq0, eq1);

        let mut eq2 = MLE::<F>::eq(&r_rev);
        eq2.reverse_bits();
        assert_eq!(eq0, eq2);

        {
            let init = MLE::new(vec![F::ONE]);
            let mut eqs = vec![init.clone()];
            r.iter().for_each(|&r| {
                let eq = eqs.last().unwrap();
                eqs.push(eq.extend_to(r));
            });
            let eq3 = eqs.last_mut().unwrap();
            eq3.reverse_bits();
            assert_eq!(eq0, *eq3);
        }
    }
}
