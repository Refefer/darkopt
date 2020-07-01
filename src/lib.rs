pub mod de;
pub mod pso;
pub mod exp;

//pub mod ga;

pub trait Fitness: Send + Sync {
    type Data;
    fn score(&self, candidate: &Self::Data) -> f32;
}

pub trait Optimizer: Clone + std::fmt::Debug + Send + Sync {
    type Stats;
    type Data: Clone + Send + Sync;

    fn fit<F: Fitness<Data=Self::Data>, FN: FnMut(Self::Stats, usize) -> ()>(
        &self,
        fit_fn: &F, 
        total_fns: usize,
        seed: u64,
        x_in: Option<&Self::Data>,
        callback: FN
    ) -> (f32, Self::Data);
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
