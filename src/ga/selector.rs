use std::fmt::Debug;

use rand::prelude::*;
use rand_distr::{Distribution,Uniform};
use float_ord::FloatOrd;

use super::Selector;

#[derive(Clone,Debug)]
pub struct Tournament(pub usize);

impl Selector for Tournament {
    fn choose<R: Rng>(&self, fitnesses: &[f32], rng: &mut R) -> usize {
        Uniform::new(0, fitnesses.len())
            .sample_iter(rng)
            .take(self.0)
            .max_by_key(|idx| FloatOrd(fitnesses[*idx]))
            .expect("K == 0 for tournament selection!")
    }
}
