use crate::{
    field::{ExtField, Field},
    transcript::{Challenge, Reader},
    utils::interpolate,
    Error,
};

pub mod algo1;
pub mod gate;
#[cfg(test)]
mod test;
pub mod zerocheck;

pub(super) fn extrapolate<F: Field, EF: ExtField<F>>(evals: &[F], target: EF) -> EF {
    let points = (0..evals.len())
        .map(|i| F::from(i as u64))
        .collect::<Vec<_>>();
    let sx = interpolate(&points, evals);
    target.horner(&sx)
}

#[tracing::instrument(level = "info", skip_all)]
pub fn reduce_sumcheck_claim<F: Field, EF: ExtField<F>, Transcript>(
    k: usize,
    degree: usize,
    transcript: &mut Transcript,
) -> Result<(EF, Vec<EF>), Error>
where
    Transcript: Reader<F> + Reader<EF> + Challenge<EF>,
{
    let (sum, v0): (F, F) = (transcript.read()?, transcript.read()?);

    let v1 = sum - v0;
    (sum == v0 + v1).then_some(()).ok_or(Error::Verify).unwrap();

    let v_rest = (0..degree - 1)
        .map(|_| transcript.read())
        .collect::<Result<Vec<_>, Error>>()?;

    let v = [v0, v1].into_iter().chain(v_rest).collect::<Vec<_>>();

    let r: EF = transcript.draw();

    let mut red: EF = extrapolate(&v, r);
    let mut rs = vec![r];

    (0..k - 1).try_for_each(|_k| {
        let v0: EF = transcript.read()?;
        let v_rest = (0..degree - 1)
            .map(|_| transcript.read())
            .collect::<Result<Vec<_>, Error>>()?;
        let v = [v0, red - v0].into_iter().chain(v_rest).collect::<Vec<_>>();
        let r = transcript.draw();

        rs.push(r);
        red = extrapolate(&v, r);
        Ok(())
    })?;

    Ok((red, rs))
}

pub fn reduce_zerocheck_claim<F: Field, EF: ExtField<F>, Transcript>(
    transcript: &mut Transcript,
    degree: usize,
    y: &[EF],
) -> Result<(EF, Vec<EF>), crate::Error>
where
    Transcript: Reader<F> + Reader<EF> + Challenge<EF>,
{
    let mut sum = EF::ZERO;
    let rs = y
        .iter()
        .enumerate()
        .map(|(round, &y)| {
            if round == 0 {
                let evals: Vec<EF> = (2..=degree)
                    .map(|_| transcript.read())
                    .collect::<Result<Vec<_>, _>>()?;

                let evals = vec![EF::ZERO; 2]
                    .into_iter()
                    .chain(evals)
                    .collect::<Vec<_>>();

                let r = transcript.draw();

                sum = extrapolate(&evals, r);
                Ok(r)
            } else {
                let evals: Vec<EF> = (0..=degree)
                    .map(|_| transcript.read())
                    .collect::<Result<Vec<_>, _>>()?;

                (sum == evals[0] * (EF::ONE - y) + evals[1] * y)
                    .then_some(())
                    .ok_or(crate::Error::Verify)?;

                let r = transcript.draw();
                sum = extrapolate(&evals, r);
                Ok(r)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((sum, rs))
}
