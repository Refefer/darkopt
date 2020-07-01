use crate::Fitness;

pub struct MatyasEnv(pub f32, pub f32);

impl Fitness for MatyasEnv {
    type Data = Vec<f32>;

    fn score(&self, candidate: &Vec<f32>) -> f32 {
        let mut x = candidate[0];
        let mut y = candidate[1];
        x += self.0;
        y += self.1;
        -(0.26 * (x.powi(2) + y.powi(2)) - 0.48 * x * y)
    }

}

pub struct RastriginEnv { pub dims: usize }

impl Fitness for RastriginEnv {
    type Data = Vec<f32>;

    fn score(&self, candidate: &Vec<f32>) -> f32 {
        let f: f32 = 10. * (self.dims as f32) + 
            candidate.iter().map(|xi| {
                (*xi).powf(2.) - 10. * (2. * std::f32::consts::PI * (*xi)).cos()
            }).sum::<f32>();
        -f
    }

}
