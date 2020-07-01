extern crate rand;

use std::fmt::Debug;
use rand::prelude::*;
use rand_distr::Distribution;

use super::Genome;

#[derive(Debug,Clone)]
pub struct Continuous<D> {
    pub dims: usize,
    pub dist: D
}

impl <D: Distribution<f32> + Send + Sync + Debug + Clone> Genome for Continuous<D> {
    type Encoded = Vec<f32>;

    fn size(&self, encode: &Self::Encoded) -> usize {
        encode.len()
    }

    fn new<R: Rng>(&self, rng: &mut R) -> Self::Encoded {
        (0..self.dims).map(|_| self.dist.sample(rng) ).collect()
    }
}
