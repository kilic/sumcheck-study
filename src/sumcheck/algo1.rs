use super::extrapolate;
use crate::{
    data::MatrixOwn,
    field::{ExtField, Field},
    transcript::{Challenge, Writer},
};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

pub mod natural {
    use super::*;

    #[tracing::instrument(level = "info", skip_all, fields(k = px.k()))]
    fn eval<F: Field, Transcript>(
        transcript: &mut Transcript,
        sum: F,
        px: &MatrixOwn<F>,
    ) -> Result<Vec<F>, crate::Error>
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
    fn fold_to_ext<F: Field, EF: ExtField<F>>(r: EF, px: &MatrixOwn<F>) -> MatrixOwn<EF> {
        let mut dst: MatrixOwn<EF> = tracing::info_span!("the alloc", k = px.k() - 1)
            .in_scope(|| MatrixOwn::zero(px.width(), px.k() - 1));
        dst.par_iter_mut()
            .zip(px.chunk_pair())
            .for_each(|(dst, (a0, a1))| {
                dst.iter_mut()
                    .zip(a0.iter())
                    .zip(a1.iter())
                    .for_each(|((dst, &a0), &a1)| *dst = r * (a1 - a0) + a0);
            });
        dst
    }

    #[tracing::instrument(level = "info", skip_all)]
    fn fold<F: Field>(r: F, src: &mut MatrixOwn<F>, dst: &mut MatrixOwn<F>) {
        assert_eq!(src.k() - 1, dst.k());
        dst.par_iter_mut()
            .zip(src.chunk_pair())
            .for_each(|(dst, (a0, a1))| {
                dst.iter_mut()
                    .zip(a0.iter())
                    .zip(a1.iter())
                    .for_each(|((dst, &a0), &a1)| *dst = r * (a1 - a0) + a0);
            });

        src.drop_hi();
        src.drop_hi();
    }

    #[tracing::instrument(level = "info", skip_all)]
    pub fn prove<F: Field, EF: ExtField<F>, Transcript>(
        sum: F,
        px: MatrixOwn<F>,
        transcript: &mut Transcript,
    ) -> Result<(EF, Vec<EF>), crate::Error>
    where
        Transcript: Writer<F> + Writer<EF> + Challenge<EF>,
    {
        let k = px.k();
        let width = px.width();
        transcript.write(sum)?;
        let mut rs = vec![];

        let (mut px, mut tmp, mut sum) = tracing::info_span!("first round").in_scope(|| {
            let evals = eval(transcript, sum, &px)?;
            let r: EF = transcript.draw();
            rs.push(r);
            let px_ext = fold_to_ext(r, &px);

            let tmp = EF::from_base_slice_parts(px.storage, px_ext.height() * px_ext.width() / 2);
            let tmp: MatrixOwn<EF> = MatrixOwn::new(width, tmp);

            Ok((px_ext, tmp, extrapolate(&evals, r)))
        })?;

        tracing::info_span!("rest").in_scope(|| {
            (1..k).try_for_each(|round| {
                let evals = eval(
                    transcript,
                    sum,
                    if round & 1 == 0 { &mut tmp } else { &mut px },
                )?;

                let r = transcript.draw();
                rs.push(r);
                sum = extrapolate(&evals, r);
                if round & 1 == 0 {
                    fold(r, &mut tmp, &mut px);
                } else {
                    fold(r, &mut px, &mut tmp);
                }
                Ok(())
            })
        })?;

        Ok((sum, rs))
    }
}

pub mod gated {
    use crate::sumcheck::gate::Gate;

    use super::*;

    #[tracing::instrument(level = "info", skip_all, fields(k = px.k()))]
    fn eval<F: Field, Transcript>(
        transcript: &mut Transcript,
        sum: F,
        px: &MatrixOwn<F>,
    ) -> Result<Vec<F>, crate::Error>
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
    fn fold_to_ext<F: Field, EF: ExtField<F>>(r: EF, px: &MatrixOwn<F>) -> MatrixOwn<EF> {
        let mut dst: MatrixOwn<EF> = tracing::info_span!("the alloc", k = px.k() - 1)
            .in_scope(|| MatrixOwn::zero(px.width(), px.k() - 1));
        dst.par_iter_mut()
            .zip(px.chunk_pair())
            .for_each(|(dst, (a0, a1))| {
                dst.iter_mut()
                    .zip(a0.iter())
                    .zip(a1.iter())
                    .for_each(|((dst, &a0), &a1)| *dst = r * (a1 - a0) + a0);
            });
        dst
    }

    #[tracing::instrument(level = "info", skip_all)]
    fn fold<F: Field>(r: F, src: &mut MatrixOwn<F>, dst: &mut MatrixOwn<F>) {
        assert_eq!(src.k() - 1, dst.k());
        dst.par_iter_mut()
            .zip(src.chunk_pair())
            .for_each(|(dst, (a0, a1))| {
                dst.iter_mut()
                    .zip(a0.iter())
                    .zip(a1.iter())
                    .for_each(|((dst, &a0), &a1)| *dst = r * (a1 - a0) + a0);
            });

        src.drop_hi();
        src.drop_hi();
    }

    #[tracing::instrument(level = "info", skip_all)]
    pub fn prove<F: Field, G: Gate, EF: ExtField<F>, Transcript>(
        sum: F,
        px: MatrixOwn<F>,
        gate: G,
        transcript: &mut Transcript,
    ) -> Result<(EF, Vec<EF>), crate::Error>
    where
        Transcript: Writer<F> + Writer<EF> + Challenge<EF>,
    {
        let k = px.k();
        let width = px.width();
        transcript.write(sum)?;
        let mut rs = vec![];

        let (mut px, mut tmp, mut sum) = tracing::info_span!("first round").in_scope(|| {
            let evals = gate.eval_cross(transcript, sum, &px)?;
            let r: EF = transcript.draw();
            rs.push(r);
            let px_ext = fold_to_ext(r, &px);

            let tmp = EF::from_base_slice_parts(px.storage, px_ext.height() * px_ext.width() / 2);
            let tmp: MatrixOwn<EF> = MatrixOwn::new(width, tmp);

            Ok((px_ext, tmp, extrapolate(&evals, r)))
        })?;

        tracing::info_span!("rest").in_scope(|| {
            (1..k).try_for_each(|round| {
                let evals = gate.eval_cross(transcript, sum, if round & 1 == 0 { &tmp } else { &px })?;
                let r = transcript.draw();
                rs.push(r);
                sum = extrapolate(&evals, r);
                if round & 1 == 0 {
                    fold(r, &mut tmp, &mut px);
                } else {
                    fold(r, &mut px, &mut tmp);
                }
                Ok(())
            })
        })?;

        Ok((sum, rs))
    }
}

pub mod reversed {
    use super::*;

    #[tracing::instrument(level = "info", skip_all)]
    fn fold_to_ext<F: Field, EF: ExtField<F>>(r: EF, px: &MatrixOwn<F>) -> MatrixOwn<EF> {
        let (lo, hi) = px.split_half();
        let mut dst: MatrixOwn<EF> = tracing::info_span!("the alloc", k = lo.k())
            .in_scope(|| MatrixOwn::zero(lo.width(), lo.k()));

        dst.par_chunks_mut(1 << 15)
            .zip(lo.par_chunks(1 << 15))
            .zip(hi.par_chunks(1 << 15))
            .for_each(|((mut dst, lo), hi)| {
                dst.iter_mut()
                    .zip(lo.iter())
                    .zip(hi.iter())
                    .for_each(|((mat, lo), hi)| {
                        mat.iter_mut()
                            .zip(lo.iter())
                            .zip(hi.iter())
                            .for_each(|((mat, &lo), &hi)| *mat = r * (hi - lo) + lo);
                    });
            });
        dst
    }

    #[tracing::instrument(level = "info", skip_all)]
    fn fold<F: Field>(r: F, px: &mut MatrixOwn<F>) {
        let (mut lo, hi) = px.split_mut_half();
        lo.par_iter_mut().zip(hi.par_iter()).for_each(|(lo, hi)| {
            lo.iter_mut()
                .zip(hi.iter())
                .for_each(|(lo, &hi)| *lo += r * (hi - *lo));
        });
        px.drop_hi();
    }

    #[tracing::instrument(level = "info", skip_all, fields(k = px.k()))]
    fn eval<F: Field, Transcript>(
        transcript: &mut Transcript,
        sum: F,
        px: &MatrixOwn<F>,
    ) -> Result<Vec<F>, crate::Error>
    where
        Transcript: Writer<F>,
    {
        let (lo, hi) = px.split_half();
        let mut evals = lo
            .par_iter()
            .zip(hi.par_iter())
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
    pub fn prove<F: Field, EF: ExtField<F>, Transcript>(
        sum: F,
        px: &MatrixOwn<F>,
        transcript: &mut Transcript,
    ) -> Result<(EF, Vec<EF>), crate::Error>
    where
        Transcript: Writer<F> + Writer<EF> + Challenge<EF>,
    {
        let k = px.k();
        transcript.write(sum)?;
        let mut rs = vec![];

        let (mut px, mut sum) = tracing::info_span!("first round").in_scope(|| {
            let evals = eval(transcript, sum, px)?;
            let r: EF = transcript.draw();
            rs.push(r);
            Ok((fold_to_ext(r, px), extrapolate(&evals, r)))
        })?;

        tracing::info_span!("rest").in_scope(|| {
            (1..k).try_for_each(|_round| {
                let evals = eval(transcript, sum, &px)?;
                let r = transcript.draw();
                rs.push(r);
                sum = extrapolate(&evals, r);
                fold(r, &mut px);
                Ok(())
            })
        })?;

        Ok((sum, rs))
    }
}
