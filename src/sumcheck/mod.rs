use crate::{
    field::{ExtField, Field},
    utils::interpolate,
};

pub mod algo1;
#[cfg(test)]
mod test;

pub(super) fn extrapolate<F: Field, EF: ExtField<F>>(evals: &[F], target: EF) -> EF {
    let points = (0..evals.len())
        .map(|i| F::from(i as u64))
        .collect::<Vec<_>>();
    let sx = interpolate(&points, evals);
    target.horner(&sx)
}
