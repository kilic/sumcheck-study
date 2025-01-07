use rayon::iter::ParallelIterator;

use crate::{data::MatrixOwn, field::Field, transcript::Writer, Error};

pub struct Source {
    index: usize,
}

impl Source {
    pub fn new(index: usize) -> Self {
        Self { index }
    }
}

impl From<usize> for Source {
    fn from(index: usize) -> Self {
        Self::new(index)
    }
}

pub trait Gate {
    fn eval_cross<F: Field, Transcript>(
        &self,
        transcript: &mut Transcript,
        sum: F,
        px: &MatrixOwn<F>,
    ) -> Result<Vec<F>, Error>
    where
        Transcript: Writer<F>;

    fn eval<F: Field>(&self, px: &MatrixOwn<F>) -> F;
}

pub struct GenericGate {
    terms: Vec<Vec<Source>>,
    degree: usize,
}

impl GenericGate {
    pub fn new(terms: Vec<Vec<Source>>, degree: usize) -> Self {
        Self { terms, degree }
    }

    fn eval_cross<F: Field>(&self, a0: &[F], a1: &[F]) -> Vec<F> {
        let mut acc_outer = vec![F::ZERO; self.degree];

        for term in self.terms.iter() {
            debug_assert!(term.len() <= self.degree);
            debug_assert!(!term.is_empty());

            let a0 = term.iter().map(|s| a0[s.index]).collect::<Vec<_>>();
            let a1 = term.iter().map(|s| a1[s.index]).collect::<Vec<_>>();

            let mut acc_inner = a1.clone();
            let dif = a1
                .iter()
                .zip(a0.iter())
                .map(|(&a1, &a0)| a1 - a0)
                .collect::<Vec<_>>();

            for (w, acc) in acc_outer.iter_mut().enumerate() {
                if w == 0 {
                    *acc += a0.iter().product::<F>();
                } else {
                    acc_inner
                        .iter_mut()
                        .zip(dif.iter())
                        .for_each(|(acc, &dif)| *acc += dif);
                    *acc += acc_inner.iter().product::<F>();
                }
            }
        }

        acc_outer
    }
}

impl Gate for GenericGate {
    #[tracing::instrument(level = "info", skip_all)]
    fn eval_cross<F: Field, Transcript>(
        &self,
        transcript: &mut Transcript,
        sum: F,
        px: &MatrixOwn<F>,
    ) -> Result<Vec<F>, Error>
    where
        Transcript: Writer<F>,
    {
        let mut evals = px
            .chunk_pair()
            .map(|(a0, a1)| self.eval_cross(a0, a1))
            .reduce(
                || vec![F::ZERO; self.degree],
                |a, b| a.iter().zip(b.iter()).map(|(&a, &b)| a + b).collect(),
            )
            .to_vec();

        evals.iter().try_for_each(|&e| transcript.write(e))?;
        evals.insert(1, sum - evals[0]);
        Ok(evals)
    }

    #[tracing::instrument(level = "info", skip_all)]
    fn eval<F: Field>(&self, px: &MatrixOwn<F>) -> F {
        px.par_iter()
            .fold(
                || F::ZERO,
                |acc, row| {
                    self.terms.iter().fold(acc, |acc, term| {
                        acc + term.iter().map(|s| row[s.index]).product::<F>()
                    })
                },
            )
            .sum::<F>()
    }
}

pub struct HandGateExample {}

impl HandGateExample {
    fn eval<F: Field>(&self, a0: &[F], a1: &[F]) -> [F; 3] {
        let v0 = a0[0] * a0[1] * a0[2];

        let dif0 = a1[0] - a0[0];
        let dif1 = a1[1] - a0[1];
        let dif2 = a1[2] - a0[2];

        let u0 = a1[0] + dif0;
        let u1 = a1[1] + dif1;
        let u2 = a1[2] + dif2;

        let v2 = u0 * u1 * u2;
        let v3 = (u0 + dif0) * (u1 + dif1) * (u2 + dif2);
        [v0, v2, v3]
    }
}

impl Gate for HandGateExample {
    #[tracing::instrument(level = "info", skip_all)]
    fn eval_cross<F: Field, Transcript>(
        &self,
        transcript: &mut Transcript,
        sum: F,
        px: &MatrixOwn<F>,
    ) -> Result<Vec<F>, Error>
    where
        Transcript: Writer<F>,
    {
        let mut evals = px
            .chunk_pair()
            .map(|(a0, a1)| self.eval(a0, a1))
            .reduce(
                || [F::ZERO, F::ZERO, F::ZERO],
                |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
            )
            .to_vec();
        evals.iter().try_for_each(|&e| transcript.write(e))?;
        evals.insert(1, sum - evals[0]);
        Ok(evals)
    }

    #[tracing::instrument(level = "info", skip_all)]
    fn eval<F: Field>(&self, px: &MatrixOwn<F>) -> F {
        px.par_iter()
            .map(|row| row.iter().product::<F>())
            .sum::<F>()
    }
}
