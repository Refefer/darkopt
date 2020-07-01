use std::marker::PhantomData;
use rand::prelude::*;
use rand_distr::{Distribution,Binomial,Uniform};

use super::Crossover;

#[derive(Copy,Clone,Debug)]
pub enum CrossoverType {
    /// Uniform over the entire genome
    Binomial,

    /// Choose one point and crosses the genom
    OnePoint,

    /// Swaps a contiguous portion of one to the other.
    TwoPoint
}

#[derive(Debug,Clone)]
pub struct Linear<A>(CrossoverType, PhantomData<A>);

impl <A> Linear<A> {
    pub fn new(ct: CrossoverType) -> Self {
        Linear(ct, PhantomData)
    }
}

impl <A: Clone + Send + Sync + std::fmt::Debug> Crossover for Linear<A> {
    type Encoded = Vec<A>;

    /// Number of parents needed for crossing
    fn parents_to_select(&self) -> usize { 2 }

    /// Creates a new offspring given a set of parents
    fn cross<R: Rng>(&self, parents: &[&Self::Encoded], rng: &mut R) -> Self::Encoded {
        let p1 = parents[0];
        let p2 = parents[1];
        let mut offspring = p1.clone();
        match self.0 {
            CrossoverType::Binomial => {
                let dist = Binomial::new(1, 0.5).expect("Should never fail!");
                let it = dist.sample_iter(rng).take(p1.len());
                it.enumerate().for_each(|(i, from_p2)| {
                    if from_p2 == 1 {
                        offspring[i] = p2[i].clone();
                    }
                });
            },
            CrossoverType::OnePoint => {
                let idx = Uniform::new(1, p1.len() - 1).sample(rng);
                offspring[idx..].clone_from_slice(&p2[idx..]);
            },
            CrossoverType::TwoPoint => {
                let start = Uniform::new(0, p1.len() - 1).sample(rng);
                let stop = Uniform::new(start + 1, p1.len()).sample(rng);
                for idx in start..stop {
                    offspring[idx] = p2[idx].clone();
                }
            }
        }
        offspring
    }

}
