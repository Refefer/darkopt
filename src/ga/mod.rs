pub mod genome;
pub mod mutator;
pub mod crossover;
pub mod selector;

pub mod fastga;

use std::fmt::Debug;
use rand::prelude::*;

/// Defines genomes
pub trait Genome: Send + Sync + Clone + Debug {
    type Encoded: Send + Sync;

    fn size(&self, encode: &Self::Encoded) -> usize;

    fn new<R: Rng>(&self, rng: &mut R) -> Self::Encoded;
}

/// Defines a mutator which can pointwise mutate a genome
pub trait Mutator: Send + Sync + Clone + Debug {
    type Encoded: Send + Sync;

    fn mutate<R: Rng>(&self, idxs: &[usize], genome: &mut Self::Encoded, rng: &mut R);
}

/// Defines a crossover method between a set of selected parents
pub trait Crossover: Send + Sync + Clone + Debug {
    type Encoded: Send + Sync + Clone;

    /// Number of parents needed for crossing
    fn parents_to_select(&self) -> usize;

    /// Creates a new offspring given a set of parents
    fn cross<R: Rng>(&self, parents: &[&Self::Encoded], rng: &mut R) -> Self::Encoded;
}

/// Selection for parents
pub trait Selector: Send + Sync + Clone + Debug {

    /// Selects a parent from the given set of indices
    fn choose<R: Rng>(&self, fitnesses: &[f32], rng: &mut R) -> usize;
}


