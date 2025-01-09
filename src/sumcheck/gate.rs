use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    data::MatrixOwn,
    field::{ExtField, Field},
    mle::MLE,
    transcript::Writer,
    Error,
};

#[derive(Debug, Clone, Copy)]
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
    fn sum_hypercube<F: Field>(&self, px: &MatrixOwn<F>) -> F;
    fn eval<F: Field, EF: ExtField<F>>(&self, rs: &[EF], px: &MatrixOwn<F>) -> EF;
}

#[derive(Debug, Clone)]
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
    fn sum_hypercube<F: Field>(&self, px: &MatrixOwn<F>) -> F {
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

    fn eval<F: Field, EF: ExtField<F>>(&self, rs: &[EF], px: &MatrixOwn<F>) -> EF {
        assert_eq!(rs.len(), px.k());

        let eq = MLE::<_>::eq(rs);
        let evals = eq
            .par_iter()
            .zip(px.par_iter())
            .map(|(&r, row)| row.iter().map(|&e| r * e).collect::<Vec<_>>())
            .reduce(
                || vec![EF::ZERO; px.width()],
                |acc, next| {
                    acc.iter()
                        .zip(next.iter())
                        .map(|(&acc, &next)| acc + next)
                        .collect::<Vec<_>>()
                },
            );

        self.terms.iter().fold(EF::ZERO, |acc, term| {
            acc + term.iter().map(|s| evals[s.index]).product::<EF>()
        })
    }
}

#[derive(Debug, Clone, Copy)]
// a*b*c
pub struct HandGateExample {}

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
            .map(|(a0, a1)| {
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
            })
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
    fn sum_hypercube<F: Field>(&self, px: &MatrixOwn<F>) -> F {
        px.par_iter()
            .map(|row| row.iter().product::<F>())
            .sum::<F>()
    }

    #[tracing::instrument(level = "info", skip_all)]
    fn eval<F: Field, EF: ExtField<F>>(&self, rs: &[EF], px: &MatrixOwn<F>) -> EF {
        assert_eq!(rs.len(), px.k());
        let eq = MLE::<_>::eq(rs);
        let evals = eq
            .par_iter()
            .zip(px.par_iter())
            .map(|(&r, row)| row.iter().map(|&e| r * e).collect::<Vec<_>>())
            .reduce(
                || vec![EF::ZERO; px.width()],
                |acc, next| {
                    acc.iter()
                        .zip(next.iter())
                        .map(|(&acc, &next)| acc + next)
                        .collect::<Vec<_>>()
                },
            );

        evals.iter().product()
    }
}

#[cfg(test)]
mod test_gate {
    use crate::{
        data::MatrixOwn,
        sumcheck::gate::{Gate, GenericGate, HandGateExample},
        utils::n_rand,
    };
    type F = crate::field::goldilocks::Goldilocks;
    type EF = crate::field::goldilocks::Goldilocks2;

    #[test]
    fn test_gate_eval() {
        let mut rng = crate::test::seed_rng();

        let k = 5;
        let d = 3;
        let n = 1 << k;
        let px = (0..d).map(|_| n_rand::<F>(&mut rng, n)).collect::<Vec<_>>();
        let px = MatrixOwn::from_columns(&px);

        let generic_gate =
            GenericGate::new(vec![vec![0usize.into(), 1usize.into(), 2usize.into()]], 3);
        let hand_gate = HandGateExample {};

        let e0 = generic_gate.sum_hypercube(&px);
        let e1 = hand_gate.sum_hypercube(&px);
        assert_eq!(e0, e1);
        let rs: Vec<EF> = n_rand(&mut rng, k);
        let e0 = generic_gate.eval(&rs, &px);
        let e1 = hand_gate.eval(&rs, &px);

        assert_eq!(e0, e1);
    }
}

pub trait ZeroGate {
    fn eval_cross_to_ext<F: Field, EF: ExtField<F>, Transcript>(
        &self,
        transcript: &mut Transcript,
        px: &MatrixOwn<F>,
        eq: &MLE<EF>,
    ) -> Result<Vec<EF>, Error>
    where
        Transcript: Writer<EF>;

    fn eval_cross<F: Field, Transcript>(
        &self,
        transcript: &mut Transcript,
        sum: F,
        px: &MatrixOwn<F>,
        eq: &MLE<F>,
        yi: F,
    ) -> Result<Vec<F>, Error>
    where
        Transcript: Writer<F>;

    fn eval<F: Field, EF: ExtField<F>>(&self, rs: &[EF], px: &MatrixOwn<F>) -> EF;
}

#[derive(Debug, Clone, Copy)]
// a*b-c
pub struct HandGateMulExample {}

impl HandGateMulExample {}

impl ZeroGate for HandGateMulExample {
    #[tracing::instrument(level = "info", skip_all)]
    fn eval_cross_to_ext<F: Field, EF: ExtField<F>, Transcript>(
        &self,
        transcript: &mut Transcript,
        px: &MatrixOwn<F>,
        eq: &MLE<EF>,
    ) -> Result<Vec<EF>, Error>
    where
        Transcript: Writer<EF>,
    {
        assert_eq!(px.k() - 1, eq.k());
        let e2 = px
            .chunk_pair()
            .zip(eq.par_iter())
            .map(|((a0, a1), &r)| {
                let u0 = a1[0].double() - a0[0];
                let u1 = a1[1].double() - a0[1];
                let u2 = a1[2].double() - a0[2];

                r * (u0 * u1 - u2)
            })
            .reduce(|| EF::ZERO, |a, b| a + b);

        transcript.write(e2)?;
        let evals = vec![EF::ZERO, EF::ZERO, e2];
        Ok(evals)
    }

    #[tracing::instrument(level = "info", skip_all)]
    fn eval_cross<F: Field, Transcript>(
        &self,
        transcript: &mut Transcript,
        sum: F,
        px: &MatrixOwn<F>,
        eq: &MLE<F>,
        yi: F,
    ) -> Result<Vec<F>, Error>
    where
        Transcript: Writer<F>,
    {
        assert_eq!(px.k() - 1, eq.k());
        let mut evals = px
            .chunk_pair()
            .zip(eq.par_iter())
            .map(|((a0, a1), &r)| {
                let e0 = r * (a0[0] * a0[1] - a0[2]);

                let u0 = a1[0].double() - a0[0];
                let u1 = a1[1].double() - a0[1];
                let u2 = a1[2].double() - a0[2];

                let e2 = r * (u0 * u1 - u2);

                [e0, e2]
            })
            .reduce(|| [F::ZERO, F::ZERO], |a, b| [a[0] + b[0], a[1] + b[1]])
            .to_vec();
        let e1 = (sum + *evals.first().unwrap() * (yi - F::ONE)) * yi.inverse().unwrap();
        evals.insert(1, e1);
        assert_eq!(evals[0] * (F::ONE - yi) + evals[1] * yi, sum);
        evals.iter().try_for_each(|&e| transcript.write(e))?;
        Ok(evals)
    }

    #[tracing::instrument(level = "info", skip_all)]
    fn eval<F: Field, EF: ExtField<F>>(&self, rs: &[EF], px: &MatrixOwn<F>) -> EF {
        assert_eq!(rs.len(), px.k());
        let eq = MLE::<_>::eq(rs);
        let evals = eq
            .par_iter()
            .zip(px.par_iter())
            .map(|(&r, row)| row.iter().map(|&e| r * e).collect::<Vec<_>>())
            .reduce(
                || vec![EF::ZERO; px.width()],
                |acc, next| {
                    acc.iter()
                        .zip(next.iter())
                        .map(|(&acc, &next)| acc + next)
                        .collect::<Vec<_>>()
                },
            );
        evals[0] * evals[1] - evals[2]
    }
}
