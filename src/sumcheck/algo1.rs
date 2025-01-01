use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use crate::{
    data::MatrixOwn,
    field::{Ext, ExtField, Extended, Field},
    transcript::{Challenge, Writer},
};

use super::extrapolate;

#[tracing::instrument(level = "info", skip_all)]
fn fold_to_ext<F: Field, EF: ExtField<F>>(r: EF, dif: MatrixOwn<F>) -> MatrixOwn<EF> {
    let (dif, acc) = dif.split_half();
    let mut dst: MatrixOwn<EF> =
        tracing::info_span!("the alloc").in_scope(|| MatrixOwn::zero(dif.width(), dif.k()));
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
    dst
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

    let e0 = lo
        .par_iter()
        .map(|next| next.iter().product::<F>())
        .reduce(|| F::ZERO, |a, b| a + b);
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

    let (mut px, mut sum) = tracing::info_span!("first round").in_scope(|| {
        let evals = eval(transcript, sum, &mut px)?;
        let r: Ext<E, F> = transcript.draw();
        rs.push(r);
        Ok((fold_to_ext(r, px), extrapolate(&evals, r)))
    })?;

    tracing::info_span!("rest").in_scope(|| {
        (1..k).try_for_each(|_round| {
            let evals = eval(transcript, sum, &mut px)?;
            let r = transcript.draw();
            rs.push(r);
            sum = extrapolate(&evals, r);
            fold(r, &mut px);
            Ok(())
        })
    })?;

    Ok((sum, rs))
}

#[tracing::instrument(level = "info", skip_all, fields(k = px.k()))]
fn eval2<F: Field, Transcript>(
    transcript: &mut Transcript,
    sum: F,
    px: &mut MatrixOwn<F>,
) -> Result<Vec<F>, crate::Error>
where
    Transcript: Writer<F>,
{
    let mut evals = px
        .chunk2()
        .map(|(a0, a1)| {
            let v0 = a0[0] * a0[1] * a0[2];
            let dif0 = a1[0] - a0[0];
            let dif1 = a1[1] - a0[1];
            let dif2 = a1[2] - a0[2];
            let u0 = a1[0] + dif0;
            let u1 = a1[1] + dif1;
            let u2 = a1[2] + dif2;
            let v2 = u0 * u1 * u2;
            let u0 = u0 + dif0;
            let u1 = u1 + dif1;
            let u2 = u2 + dif2;
            let v3 = u0 * u1 * u2;
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
fn fold_to_ext2<F: Field, EF: ExtField<F>>(r: EF, px: MatrixOwn<F>) -> MatrixOwn<EF> {
    let mut dst: MatrixOwn<EF> =
        tracing::info_span!("the alloc").in_scope(|| MatrixOwn::zero(px.width(), px.k() - 1));
    dst.par_iter_mut()
        .zip(px.chunk2())
        .for_each(|(dst, (a0, a1))| {
            dst.iter_mut()
                .zip(a0.iter())
                .zip(a1.iter())
                .for_each(|((dst, &a0), &a1)| *dst = r * (a1 - a0) + a0);
        });
    dst
}

#[tracing::instrument(level = "info", skip_all)]
fn fold2<F: Field>(r: F, px: &mut MatrixOwn<F>) {
    let mut tmp: MatrixOwn<F> =
        tracing::info_span!("tmp alloc").in_scope(|| MatrixOwn::zero(px.width(), px.k() - 1));
    tmp.par_iter_mut()
        .zip(px.chunk2())
        .for_each(|(dst, (a0, a1))| {
            dst.iter_mut()
                .zip(a0.iter())
                .zip(a1.iter())
                .for_each(|((dst, &a0), &a1)| *dst = r * (a1 - a0) + a0);
        });
    px.par_iter_mut()
        .zip(tmp.par_iter())
        .for_each(|(px, tmp)| px.copy_from_slice(tmp));

    px.drop_hi();
}

#[tracing::instrument(level = "info", skip_all)]
pub fn prove2<const E: usize, F: Extended<E>, Transcript>(
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

    let (mut px, mut sum) = tracing::info_span!("first round").in_scope(|| {
        let evals = eval2(transcript, sum, &mut px)?;
        let r: Ext<E, F> = transcript.draw();
        rs.push(r);
        Ok((fold_to_ext2(r, px), extrapolate(&evals, r)))
    })?;

    tracing::info_span!("rest").in_scope(|| {
        (1..k).try_for_each(|_round| {
            let evals = eval2(transcript, sum, &mut px)?;
            let r = transcript.draw();
            rs.push(r);
            sum = extrapolate(&evals, r);
            fold2(r, &mut px);
            Ok(())
        })
    })?;

    Ok((sum, rs))
}

#[tracing::instrument(level = "info", skip_all)]
fn fold_to_ext3<F: Field, EF: ExtField<F>>(r: EF, px: MatrixOwn<F>) -> MatrixOwn<EF> {
    let (lo, hi) = px.split_half();
    let mut dst: MatrixOwn<EF> =
        tracing::info_span!("the alloc").in_scope(|| MatrixOwn::zero(lo.width(), lo.k()));
    dst.par_iter_mut()
        .zip(lo.par_iter())
        .zip(hi.par_iter())
        .for_each(|((mat, lo), hi)| {
            mat.iter_mut()
                .zip(lo.iter())
                .zip(hi.iter())
                .for_each(|((mat, &lo), &hi)| *mat = r * (hi - lo) + lo);
        });
    dst
}

#[tracing::instrument(level = "info", skip_all)]
fn fold3<F: Field>(r: F, px: &mut MatrixOwn<F>) {
    let (mut lo, hi) = px.split_mut_half();
    lo.par_iter_mut().zip(hi.par_iter()).for_each(|(lo, hi)| {
        lo.iter_mut()
            .zip(hi.iter())
            .for_each(|(lo, &hi)| *lo += r * (hi - *lo));
    });
    px.drop_hi();
}

#[tracing::instrument(level = "info", skip_all, fields(k = px.k()))]
fn eval3<F: Field, Transcript>(
    transcript: &mut Transcript,
    sum: F,
    px: &mut MatrixOwn<F>,
) -> Result<Vec<F>, crate::Error>
where
    Transcript: Writer<F>,
{
    let (lo, hi) = px.split_mut_half();

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
pub fn prove3<const E: usize, F: Extended<E>, Transcript>(
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

    let (mut px, mut sum) = tracing::info_span!("first round").in_scope(|| {
        let evals = eval3(transcript, sum, &mut px)?;
        let r: Ext<E, F> = transcript.draw();
        rs.push(r);
        Ok((fold_to_ext3(r, px), extrapolate(&evals, r)))
    })?;

    tracing::info_span!("rest").in_scope(|| {
        (1..k).try_for_each(|_round| {
            let evals = eval3(transcript, sum, &mut px)?;
            let r = transcript.draw();
            rs.push(r);
            sum = extrapolate(&evals, r);
            fold3(r, &mut px);
            Ok(())
        })
    })?;

    Ok((sum, rs))
}
