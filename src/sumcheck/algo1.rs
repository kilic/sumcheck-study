use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use crate::{
    data::MatrixOwn,
    field::{Ext, ExtField, Extended, Field},
    transcript::{Challenge, Writer},
};

use super::extrapolate;

#[tracing::instrument(level = "info", skip_all)]
fn fold_to_ext<F: Field, EF: ExtField<F>>(r: EF, dif: MatrixOwn<F>, dst: &mut MatrixOwn<EF>) {
    let (dif, acc) = dif.split_half();
    let r_shift = EF::from(dif.width() as u64) - r;
    dst.par_iter_mut()
        .zip(acc.par_iter())
        .zip(dif.par_iter())
        .for_each(|((mat, acc), dif)| {
            mat.iter_mut()
                .zip(acc.iter())
                .zip(dif.iter())
                .for_each(|((mat, &acc), &dif)| *mat = r_shift * dif + acc);
        });
}

#[tracing::instrument(level = "info", skip_all)]
fn fold<F: Field>(r: F, px: &mut MatrixOwn<F>) {
    let (mut dif, acc) = px.split_mut_half();
    let r_shift = F::from(dif.width() as u64) - r;
    dif.par_iter_mut()
        .zip(acc.par_iter())
        .for_each(|(dif, acc)| {
            dif.iter_mut()
                .zip(acc.iter())
                .for_each(|(dif, &acc)| *dif = r_shift * *dif + acc);
        });
    px.drop_hi();
}

#[tracing::instrument(level = "info", skip_all, fields(k = px.k()))]
fn eval<F: Field, Transcript>(
    transcript: &mut Transcript,
    sum: F,
    px: &mut MatrixOwn<F>,
) -> Result<Vec<F>, crate::Error>
where
    Transcript: Writer<F>,
{
    let n = px.width();
    let (mut lo, mut hi) = px.split_mut_half();

    let e0 = tracing::info_span!("e0").in_scope(|| {
        lo.par_iter()
            .map(|next| next.iter().product::<F>())
            .reduce(|| F::ZERO, |a, b| a + b)
    });
    transcript.write(e0)?;
    let mut evals = vec![e0, sum - e0];

    #[cfg(debug_assertions)]
    {
        let e1 = hi
            .par_iter()
            .map(|next| next.iter().product::<F>())
            .reduce(|| F::ZERO, |a, b| a + b);
        assert_eq!(e1, *evals.last().unwrap());
    }

    lo.par_iter_mut()
        .zip(hi.par_iter())
        .for_each(|(lo, hi)| lo.iter_mut().zip(hi.iter()).for_each(|(lo, &hi)| *lo -= hi));

    tracing::info_span!("ei").in_scope(|| {
        for _ in 2..=n {
            let ei = lo
                .par_iter()
                .zip(hi.par_iter_mut())
                .map(|(lo, hi)| {
                    hi.iter_mut().zip(lo.iter()).for_each(|(hi, &lo)| *hi -= lo);
                    hi.iter().product::<F>()
                })
                .reduce(|| F::ZERO, |a, b| a + b);
            transcript.write(ei)?;
            evals.push(ei);
        }
        Ok(())
    })?;

    Ok(evals)
}

#[tracing::instrument(level = "info", skip_all)]
pub fn prove<const E: usize, F: Extended<E>, Transcript>(
    sum: F,
    mut px: MatrixOwn<F>,
    transcript: &mut Transcript,
) -> Result<(Ext<E, F>, Vec<Ext<E, F>>), crate::Error>
where
    Transcript: Writer<F> + Writer<Ext<E, F>> + Challenge<Ext<E, F>>,
{
    let k = px.k();
    transcript.write(sum)?;
    let mut rs = vec![];
    let width = px.width();
    let mut px_ext: MatrixOwn<Ext<E, F>> = MatrixOwn::zero(width, px.k() - 1);

    let (mut _px_ext, mut sum) = tracing::info_span!("first round").in_scope(|| {
        let evals = eval(transcript, sum, &mut px)?;
        let r: Ext<E, F> = transcript.draw();
        rs.push(r);
        Ok((fold_to_ext(r, px, &mut px_ext), extrapolate(&evals, r)))
    })?;

    tracing::info_span!("rest").in_scope(|| {
        (1..k).try_for_each(|_round| {
            if _round == 1 {
                tracing::info_span!("round1").in_scope(|| {
                    let evals = eval(transcript, sum, &mut px_ext)?;
                    let r = transcript.draw();
                    sum = extrapolate(&evals, r);
                    fold(r, &mut px_ext);
                    rs.push(r);
                    Ok(())
                })
            } else {
                let evals = eval(transcript, sum, &mut px_ext)?;
                let r = transcript.draw();
                sum = extrapolate(&evals, r);
                fold(r, &mut px_ext);
                rs.push(r);
                Ok(())
            }
        })
    })?;

    Ok((sum, rs))
}

#[tracing::instrument(level = "info", skip_all)]
pub fn prove2<F: Field, EF: ExtField<F>, Transcript>(
    sum: F,
    mut px: MatrixOwn<F>,
    transcript: &mut Transcript,
) -> Result<(EF, Vec<EF>), crate::Error>
where
    Transcript: Writer<F> + Writer<EF> + Challenge<EF>,
{
    let k = px.k();
    transcript.write(sum)?;
    let mut rs = vec![];
    let width = px.width();
    let mut px_ext: MatrixOwn<EF> = MatrixOwn::zero(width, px.k() - 1);

    let (mut _px_ext, mut sum) = tracing::info_span!("first round").in_scope(|| {
        let evals = eval(transcript, sum, &mut px)?;
        let r = transcript.draw();
        rs.push(r);
        Ok((fold_to_ext(r, px, &mut px_ext), extrapolate(&evals, r)))
    })?;

    tracing::info_span!("rest").in_scope(|| {
        (1..k).try_for_each(|_round| {
            if _round == 1 {
                tracing::info_span!("round1").in_scope(|| {
                    let evals = eval(transcript, sum, &mut px_ext)?;
                    let r = transcript.draw();
                    sum = extrapolate(&evals, r);
                    fold(r, &mut px_ext);
                    rs.push(r);
                    Ok(())
                })
            } else {
                let evals = eval(transcript, sum, &mut px_ext)?;
                let r = transcript.draw();
                sum = extrapolate(&evals, r);
                fold(r, &mut px_ext);
                rs.push(r);
                Ok(())
            }
        })
    })?;

    Ok((sum, rs))
}
