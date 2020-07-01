extern crate rand;
extern crate rayon;
extern crate float_ord;

use float_ord::FloatOrd;

use rand::prelude::*;
use rand_distr::{Distribution,Normal,Uniform};
use rayon::prelude::*;

use crate::{Fitness,Optimizer};

#[derive(Clone,Copy,Debug)]
pub enum CrossoverType {
    TwoPoint,
    Uniform(f32)
}

#[derive(Clone,Copy,Debug)]
pub struct DePlus<D> {
    /// Input space
    pub dims: usize,

    /// Population size.
    pub lambda: usize,

    /// Dithered learning rate, F.  A good default is (0.1, 0.9)
    pub f: (f32, f32),

    /// Crossover type
    pub cr: CrossoverType,

    /// Likelihood of replacing a candidate with a randomly perturb best.  0.1 is a good
    /// default.
    pub m: f32,

    /// Expansion constant for random perturbation.  A good value is between 2 to 5
    pub exp: f32,

    /// If enabled, restarts around the best value if it doesn't improve after 
    /// K iterations.  0 means turn off completely
    pub polish_on_stale: Option<usize>,

    /// If enabled, fully restarts the job with new random values.  0 means turn off.
    pub restart_on_stale: Option<usize>,

    /// Distribution to sample from for initialization
    pub init_dist: D
}

impl <D: Distribution<f64> + Sync + Clone> DePlus<D> {

    fn run_pass<F: Fitness<Data=Vec<f32>>, FN: FnMut(f32, usize) -> ()>(
        &self, 
        fit_fn: &F, 
        total_fns: usize, 
        seed: u64, 
        x_in: Option<&Vec<f32>>,
        callback: &mut FN
    ) -> (usize, f32, Vec<f32>) {

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Initialize population
        let mut pop: Vec<_> = (0..self.lambda).map(|_| {
            let mut v = vec![0.; self.dims];
            v.iter_mut().for_each(|vi| *vi = self.init_dist.sample(&mut rng) as f32);
            v
        }).collect();

        // If an x_in has been provided, offset the population by it
        if let Some(x_in_v) = x_in {
            pop.iter_mut().enumerate().for_each(|(i, p)| {
                p.iter_mut().zip(x_in_v.iter()).for_each(|(pi, vi)| {
                    // First one gets x_in as the basis
                    if i == 0 {
                        *pi = *vi;
                    } else {
                        *pi += vi;
                    }
                });
            });
        }

        let mut tmp_pop = pop.clone();

        // Generate initial fitnesses
        let mut fits: Vec<_> = pop.iter().map(|p| fit_fn.score(&p)).collect();

        // Initial function counts
        let mut fns = self.lambda;

        let norm_dist = Normal::new(0.0, 1.0).unwrap();

        // Tracks time since last update of the best candidate.  We use this to
        // determine polishing.
        let mut best_fit = std::f32::NEG_INFINITY;
        let mut stale_len = 0;

        // Checks how long until the next update
        let mut last_update = 0;

        let early_terminate = self.restart_on_stale.unwrap_or(0);
        let pos = self.polish_on_stale.unwrap_or(0);
        while fns < total_fns {
            // Get the best candidate
            let best_idx = (0..self.lambda).max_by_key(|i| FloatOrd(fits[*i]))
                .expect("Should never be empty!");

            // Check if we've improved
            if fits[best_idx] > best_fit {
                best_fit = fits[best_idx];
                stale_len = 0;
                last_update = 0;
            }

            // check if we're early terminating
            if early_terminate > 0 && last_update == early_terminate {
                break
            }

            if pos > 0 && stale_len == pos {

                let best = pop[best_idx].clone();

                // Re build population, preserving the best one
                pop.iter_mut().enumerate().for_each(|(i, p)| {
                    if i != best_idx {
                        p.iter_mut().zip(best.iter())
                            .for_each(|(vi, bi)| *vi = *bi + self.init_dist.sample(&mut rng) as f32);
                    }
                });

                // Recompute the fits, sans the best
                fits.iter_mut().enumerate().for_each(|(i, f)| {
                    if i != best_idx {
                        *f = fit_fn.score(&pop[i]);
                    }
                });
                fns += pop.len() - 1;
                stale_len = 0;
                continue
            }

            stale_len += 1;
            last_update += 1;

            let best = &pop[best_idx];

            let f_lr = Uniform::new(self.f.0, self.f.1).sample(&mut rng);

            // Generate mutation vector
            let uniform = Uniform::new(0., 1.);
            tmp_pop.par_iter_mut().zip(fits.par_iter_mut())
                    .enumerate().for_each(|(idx, (x, f))| {

                let orig_x = &pop[idx];
                let mut local_rng = rand::rngs::StdRng::seed_from_u64(
                    seed + (idx + fns) as u64);

                // Randomize a candidate in the population
                if idx != best_idx && uniform.sample(&mut local_rng) < self.m {

                    // Figure out magnitude between current x and the best
                    x.iter_mut().zip(orig_x.iter()).enumerate().for_each(|(i, (xi, oxi))| {
                        *xi = oxi - best[i];
                    });

                    let orig_mag = l2norm(&x);

                    // Ok, generate new vector
                    x.iter_mut().for_each(|xi| {
                        *xi = norm_dist.sample(&mut local_rng) as f32;
                    });

                    // Normalize it
                    let v_mag = l2norm(&x);

                    // Expand it out within the expansion radius
                    x.iter_mut().enumerate().for_each(|(i, xi)| {
                        *xi = best[i] + orig_mag * self.exp * (*xi) / v_mag;
                    });

                    // Just override the fitness
                    let new_f = fit_fn.score(&x);
                    if new_f.is_finite() {
                        *f = new_f;
                    } else {
                        x.iter_mut().zip(orig_x.iter()).for_each(|(xi, oxi)| {
                            *xi = *oxi;
                        });
                    }


                } else {
                    // x_b +  F * (x_a - x_b)

                    // Select two candidates for the combination
                    let mut slice = pop.choose_multiple(&mut local_rng, 2);
                    let a = slice.next().expect("Number of candidates to low!");
                    let b = slice.next().expect("Number of candidates to low!");

                    match self.cr {
                        CrossoverType::TwoPoint => {
                            let d = Uniform::new(0, self.dims);
                            let mut start = d.sample(&mut local_rng);
                            let end       = d.sample(&mut local_rng);
                            if start == end {
                                start = (end + 1) % self.dims;
                            }
                            x.copy_from_slice(orig_x.as_slice());
                            while start != end {
                                x[start] = best[start] + f_lr * (a[start] - b[start]);
                                start = (start + 1) % self.dims;
                            }
                        },
                        
                        CrossoverType::Uniform(cr) => {
                    
                            // Generate new vector
                            x.iter_mut().enumerate().for_each(|(i, xi)| {
                                // If we are mutating
                                if uniform.sample(&mut local_rng) < cr {
                                    *xi = best[i] + f_lr * (a[i] - b[i]);
                                } else {
                                    *xi = orig_x[i];
                                }
                            });
                        }
                    }

                    // Score the new individual
                    let new_fitness = fit_fn.score(&x);
                    if new_fitness.is_finite() && new_fitness > *f {
                        *f = new_fitness;
                    } else {
                        // Copy over the original x
                        x.iter_mut().zip(orig_x.iter()).for_each(|(xi, oxi)| {
                            *xi = *oxi;
                        });
                    }
                }
            });

            // Swap tmp with orig
            std::mem::swap(&mut pop, &mut tmp_pop);
            fns += self.lambda;

            // Callback
            callback({
                let best_idx = (0..self.lambda).max_by_key(|i| FloatOrd(fits[*i]))
                    .expect("Should never be empty!");
                fits[best_idx]
            }, if fns < total_fns {total_fns - fns} else {0});
        }

        // Get the best candidate!
        let best_idx = (0..self.lambda).max_by_key(|i| FloatOrd(fits[*i]))
            .expect("Should never be empty!");

        assert!(fits[best_idx].is_finite(), 
                format!("IDX: {}, vec: {:?}", best_idx, pop[best_idx]));

        (fns, fits[best_idx], pop.swap_remove(best_idx))
    }
}

impl <D: Distribution<f64> + Sync + Send + Clone + std::fmt::Debug> Optimizer for DePlus<D> {

    type Stats = f32;
    type Data = Vec<f32>;

    fn fit<F: Fitness<Data=Self::Data>, FN: FnMut(f32, usize) -> ()>(
        &self, 
        fit_fn: &F, 
        total_fns: usize, 
        seed: u64, 
        x_in: Option<&Vec<f32>>,
        mut callback: FN
    ) -> (f32, Vec<f32>) {

        let mut fits = 0;
        let mut best_fit = std::f32::NEG_INFINITY;
        let mut best_cand = vec![0.; self.dims];
        while fits < total_fns {
            let (fn_rem, fit, cand) = self.run_pass(fit_fn, total_fns, 
                          seed + fits as u64, x_in, 
                          &mut callback);

            fits += fn_rem;
            if fit > best_fit {
                best_fit = fit;
                best_cand = cand;
            }
        }

        (best_fit, best_cand)
    }

}

fn l2norm(v: &[f32]) -> f32 {
    v.iter()
        .map(|vi| vi.powi(2))
        .sum::<f32>()
        .powf(0.5)
}

#[cfg(test)]
mod test_de {
    use super::*;
    use rand_distr::StandardNormal;
    use crate::exp::*;

    #[test]
    fn test_matyas() {
        let de = DePlus {
            dims: 2,
            lambda: 30,
            f: (0.1, 1.),
            cr: CrossoverType::Uniform(0.9),
            m: 0.1,
            exp: 3.,
            polish_on_stale: None,
            restart_on_stale: None,
            init_dist: StandardNormal
        };

        let fit_fn = MatyasEnv(-10., 10.);
        let (fit, results) = de.fit(&fit_fn, 10000, 2020, None, |_best_fit, _fns_remaining| {});
        assert_eq!(fit, 0.);
        assert_eq!(results[0], 10.);
        assert_eq!(results[1], -10.);
    }
    

}
