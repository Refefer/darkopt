extern crate rand;
extern crate rayon;
extern crate float_ord;

use std::fmt::Debug;
use std::marker::PhantomData;
use float_ord::FloatOrd;

use rand::prelude::*;
use rand_distr::Uniform;
use rayon::prelude::*;

use super::*;
use crate::{Fitness,Optimizer};

#[derive(Clone,Copy,Debug)]
pub struct FastGA<G,M,CO,S,E> {
    /// Population size.
    pub lambda: usize,

    /// Whether to ensure the best candidates persist
    pub elitism: bool,

    pub max_mutate: f32,

    pub genome: G,

    pub mutator: M,

    pub cross_over: CO,

    pub selector: S,

    encoded: PhantomData<E>
}

impl <E: Send + Sync + Clone + Debug,
      G: Genome<Encoded=E>, 
      M: Mutator<Encoded=E>, 
      CO: Crossover<Encoded=E>,
      S: Selector> FastGA<G,M,CO,S,E> {

//impl <G,M,CO,S,E> FastGA<G,M,CO,S,E> {

    pub fn new(
        lambda: usize, 
        elitism: bool, 
        max_mutate: f32,
        genome: G, 
        mutator: M, 
        cross_over: CO,
        selector: S
    ) -> Self {
        FastGA {
            lambda,
            elitism,
            max_mutate,
            genome,
            mutator,
            cross_over,
            selector,
            encoded: PhantomData
        }
    }
    
}
   
impl <E: Send + Sync + Clone + Debug,
      G: Genome<Encoded=E>, 
      M: Mutator<Encoded=E>, 
      CO: Crossover<Encoded=E>,
      S: Selector> Optimizer for FastGA<G,M,CO,S,E> {
    type Stats = f32;
    type Data = E;

    fn fit<F: Fitness<Data=E>, FN: FnMut(f32, usize) -> ()>(
        &self, 
        fit_fn: &F, 
        total_fns: usize, 
        seed: u64, 
        x_in: Option<&E>,
        mut callback: FN
    ) -> (f32, E) {

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Create initial genome set
        let mut parents: Vec<_> = (0..self.lambda).map(|_| {
            self.genome.new(&mut rng)
        }).collect();

        if let Some(x_in) = x_in {
            parents[0] = x_in.clone();
        }

        let mut fns = self.lambda;

        let mut fitness: Vec<_> = parents.iter().map(|p| {
            fit_fn.score(p)
        }).collect();

        // Create copy spot
        let mut children = parents.clone();
        while fns < total_fns {
            let best_idx = (0..parents.len())
                .max_by_key(|idx| FloatOrd(fitness[*idx]))
                .expect("lambda needs to be greater than zero!");

            callback(fitness[best_idx], total_fns - fns);

            // Generate new seed
            let new_seed: u64 = rng.sample(Uniform::new(0, 1<<63));
            children.par_iter_mut().enumerate().for_each(|(i, c)| {
                // Initialize new thread seed
                let mut local_rng = rand::rngs::StdRng::seed_from_u64(new_seed + i as u64);

                // Grab the number of parents of interest
                let parents: Vec<_> = (0..self.cross_over.parents_to_select()).map(|_| {
                    &parents[self.selector.choose(&fitness, &mut local_rng)]
                }).collect();

                // Breed new child
                let mut child = self.cross_over.cross(&parents, &mut local_rng);

                // Figure out the mutation points
                let num_genes = self.genome.size(&child);
                let max_dim = ((self.max_mutate * num_genes as f32) as usize).max(2);
                let mut p: Vec<_> = (1..max_dim)
                    .map(|m| 1. / (m as f32).powf(1.5))
                    .collect();
                let total = p.iter().sum::<f32>();
                p.iter_mut().for_each(|mi| *mi /= total);
                let choices: Vec<_> = (1..max_dim).collect();
                let to_mutate = choices
                    .choose_weighted(&mut local_rng, |i| p[*i - 1])
                    .expect("Have an empty genome!  Unsupported");

                let indices: Vec<_> = Uniform::new(0, num_genes)
                    .sample_iter(&mut local_rng)
                    .take(*to_mutate)
                    .collect();

                // Mutate the child
                self.mutator.mutate(&indices, &mut child, &mut local_rng);
                std::mem::swap(c, &mut child);
            });

            // Compute new fitness for the children
            let mut children_fit: Vec<_> = children.par_iter()
                .map(|c| fit_fn.score(c))
                .collect();

            // If elitism is on, copy the best parent over from the previous
            // generation
            if self.elitism {
                children.push(parents[best_idx].clone());
                children_fit.push(fitness[best_idx]);
            }

            // Swap children and parents
            std::mem::swap(&mut children, &mut parents);
            fitness = children_fit;
            while children.len() > self.lambda {
                children.pop();
            }
            fns += self.lambda;
        }

        let best_idx = (0..parents.len())
                .max_by_key(|idx| FloatOrd(fitness[*idx]))
                .expect("lambda needs to be greater than zero!");

        (fitness[best_idx], parents.swap_remove(best_idx))
    }

}

#[cfg(test)]
mod test_ga {
    use super::*;
    use rand_distr::Normal;
    use crate::exp::*;
    use crate::ga::selector::*;
    use crate::ga::mutator::*;
    use crate::ga::genome::*;
    use crate::ga::crossover::*;

    #[test]
    fn test_matyas() {
        let opt: FastGA<_,_,_,_,Vec<f32>> = FastGA::new(
            100,
            true,
            1.,
            Continuous { dims: 2, dist: Normal::new(0., 1f32).unwrap() },
            ContinuousMutator(Normal::new(0., 1f32).unwrap()),           
            Linear::new(CrossoverType::TwoPoint),
            Tournament(4));

        let fit_fn = MatyasEnv(-10., 10.);
        let (fit, results) = opt.fit(&fit_fn, 10000, 2020, None, 
                                     |_best_fit, _fns_remaining| {println!("best_fit: {}", _best_fit);});
        assert!(fit.abs() < 1e-5);
        assert!((results[0] - 10.).abs() < 1e-2);
        assert!((results[1] + 10.).abs() < 1e-2);
    }
    

}
