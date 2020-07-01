use std::fmt::Debug;

use rand::prelude::*;
use rand_distr::Distribution;

use super::Mutator;

#[derive(Debug,Clone)]
pub struct ContinuousMutator<D>(pub D);

impl <D: Distribution<f32> + Send + Sync + Clone + std::fmt::Debug> Mutator for ContinuousMutator<D> {
    type Encoded = Vec<f32>;

    fn mutate<R: Rng>(&self, idxs: &[usize], genome: &mut Self::Encoded, rng: &mut R) {
        for idx in idxs {
            genome[*idx] += rng.sample(&self.0);
        }
    }

}
