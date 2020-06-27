extern crate rand;
extern crate rayon;
extern crate float_ord;

use float_ord::FloatOrd;

use rand::prelude::*;
use rand_distr::{Distribution,Normal,Uniform};
use rayon::prelude::*;

use crate::{Fitness,Optimizer};

#[derive(Clone,Debug)]
struct Particle {
    position: Vec<f32>,
    velocity: Vec<f32>,
    best_seen: Vec<f32>,
    fitness: f32,
    best_fitness: f32
}

impl Particle {
    fn new<D: Distribution<f32>, R: Rng>(dims: usize, mut rng: R, d: &D) -> Self {
        Particle {
            position: d.sample_iter(&mut rng).take(dims).collect(),
            velocity: vec![0.; dims],
            best_seen: vec![0.; dims],
            fitness: std::f32::MIN,
            best_fitness: std::f32::MIN
        }
    }

    fn evaluate<F: Fitness>(&mut self, f: &F) -> f32 {
        self.fitness = f.score(&self.position);
        if self.fitness > self.best_fitness {
            self.best_seen.copy_from_slice(&self.position);
            self.best_fitness = self.fitness;
        }
        self.fitness
    }

    fn update<R: Rng>(
        &mut self, 
        global_best: &[f32], 
        w: f32, 
        c_1: f32, 
        c_2: f32, 
        mut rng: R) 
    {
        let d = Uniform::new(0., 1.);
        
        let p = &self.position;
        let bs = &self.best_seen;
        self.velocity.iter_mut().enumerate().for_each(|(i, vi)| {
            let r_1 = d.sample(&mut rng);
            let r_2 = d.sample(&mut rng);
            *vi = w * *vi + 
                r_1 * c_1 * (global_best[i] - p[i]) +
                r_2 * c_2 * (bs[i] - p[i]);
        });

        self.position.iter_mut().zip(self.velocity.iter()).for_each(|(pi, vi)| {
            *pi += vi;
        });
    }
}

#[derive(Clone,Debug)]
pub struct PSO {
    /// Number of dimensions in the genome
    dims: usize,
    /// Number of particles in the swarm
    swarm_size: usize,

    /// Momentum coefficient
    w: f32,

    /// Global bias coefficient
    c_1: f32,

    /// Local bias coefficient
    c_2: f32
}

impl PSO {
    
    fn get_best(swarm: &[Particle], cur_best: &mut [f32]) -> f32 {
        // Get best candidate
        let best_p = swarm.iter()
            .max_by_key(|p| FloatOrd(p.best_fitness))
            .expect("Swarm should always be at least a size of one");
        cur_best.copy_from_slice(&best_p.best_seen);
        best_p.best_fitness
    }
}

impl Optimizer for PSO {

    fn fit<F: Fitness, FN: FnMut(f32, usize) -> ()>(
        &self,
        fit_fn: &F,
        total_fns: usize,
        seed: u64,
        x_in: Option<&[f32]>,
        mut callback: FN
    ) -> (f32, Vec<f32>) {

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dist = Normal::new(0., 1.).unwrap();
        // Initialize swarm
        let mut swarm: Vec<_> = (0..self.swarm_size).map(|i| {
            let mut p = Particle::new(self.dims, &mut rng, &dist);
            if i == 0 && x_in.is_some() {
                if let Some(v) = x_in {
                    p.position.copy_from_slice(&v);
               }
            }

            p.evaluate(fit_fn);
            p
        }).collect();

        let mut fn_count = self.swarm_size;

        let mut rngs = (&mut rng)
            .sample_iter(Uniform::new(0, std::u64::MAX))
            .take(self.swarm_size)
            .map(|seed| rand::rngs::StdRng::seed_from_u64(seed))
            .collect::<Vec<_>>();
 
        let mut global_best = vec![0.; self.dims];

        while (fn_count + self.swarm_size) < total_fns {
            let global_fit = PSO::get_best(&swarm, &mut global_best);
            callback(global_fit, total_fns - fn_count);

            swarm.par_iter_mut().zip(rngs.par_iter_mut()).for_each(|(p, lrng)| {
                p.update(&global_best, self.w, self.c_1, self.c_2, lrng);
                p.evaluate(fit_fn);
            });
            fn_count += swarm.len();
        }

        let global_fit = PSO::get_best(&swarm, &mut global_best);

        (global_fit, global_best)
    }
}


#[cfg(test)]
mod test_pso {
    use super::*;
    use crate::exp::MatyasEnv;

    #[test]
    fn test_matyas() {
        let de = PSO {
            dims: 2,
            swarm_size: 30,
            w: 0.8,
            c_1: 0.5,
            c_2: 1.
        };

        let fit_fn = MatyasEnv(-10., 10.);
        let (fit, results) = de.fit(&fit_fn, 10000, 2020, None, 
                                    |_best_fit, _fns_remaining| {});
        assert_eq!(fit, 0.);
        assert_eq!(results[0], 10.);
        assert_eq!(results[1], -10.);
    }
}
