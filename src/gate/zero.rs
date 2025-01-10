use crate::{
    data::MatrixOwn,
    field::{ExtField, Field},
    mle::MLE,
    transcript::Writer,
    Error,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

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
