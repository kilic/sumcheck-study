use super::{extrapolate, gate::ZeroGate};
use crate::mle::MLE;
use crate::{
    data::MatrixOwn,
    field::{ExtField, Field},
    transcript::{Challenge, Writer},
};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

#[tracing::instrument(level = "info", skip_all)]
pub fn make_eqs<F: Field>(ys: &[F]) -> Vec<(MLE<F>, F)> {
    let init = MLE::new(vec![F::ONE]);
    let mut eqs = vec![(init, *ys.last().unwrap())];
    ys.iter().rev().skip(1).for_each(|&ycur| {
        let (eq, yprev) = eqs.last().unwrap();
        eqs.push((eq.extend_to(*yprev), ycur));
    });
    eqs
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
pub fn prove<F: Field, EF: ExtField<F>, G: ZeroGate, Transcript>(
    transcript: &mut Transcript,
    px: MatrixOwn<F>,
    gate: G,
    y: &[EF],
) -> Result<(EF, Vec<EF>), crate::Error>
where
    Transcript: Writer<F> + Writer<EF> + Challenge<EF>,
{
    let k = px.k();
    let width = px.width();
    let mut rs = vec![];

    let mut eqs = make_eqs(y);
    eqs.reverse();

    let (mut px, mut tmp, mut sum1) = tracing::info_span!("first round").in_scope(|| {
        let (eq2, _yi) = &eqs[0];

        let evals = gate.eval_cross_to_ext(transcript, &px, eq2)?;

        let r: EF = transcript.draw();
        rs.push(r);

        let px_ext = fold_to_ext(r, &px);

        let tmp = EF::from_base_slice_parts(px.storage, px_ext.height() * px_ext.width() / 2);
        let tmp: MatrixOwn<EF> = MatrixOwn::new(width, tmp);
        Ok((px_ext, tmp, extrapolate(&evals, r)))
    })?;

    tracing::info_span!("rest").in_scope(|| {
        (1..k).try_for_each(|round| {
            let (eq2, yi) = &eqs[round];

            let evals = gate.eval_cross(
                transcript,
                sum1,
                if round & 1 == 0 { &tmp } else { &px },
                eq2,
                *yi,
            )?;

            let r = transcript.draw();
            rs.push(r);
            sum1 = extrapolate(&evals, r);
            if round & 1 == 0 {
                fold(r, &mut tmp, &mut px);
            } else {
                fold(r, &mut px, &mut tmp);
            }
            Ok(())
        })
    })?;
    Ok((sum1, rs))
}
