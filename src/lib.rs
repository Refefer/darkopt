pub mod de;
pub mod pso;
pub mod exp;

pub trait Fitness: Send + Sync {
    fn score(&self, candidate: &[f32]) -> f32;
}

pub trait Optimizer: Clone + std::fmt::Debug + Send + Sync {
   fn fit<F: Fitness, FN: FnMut(f32, usize) -> ()>(
        &self,
        fit_fn: &F, 
        total_fns: usize,
        seed: u64,
        x_in: Option<&[f32]>,
        callback: FN
    ) -> (f32, Vec<f32>);
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
