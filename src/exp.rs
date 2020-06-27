use crate::Fitness;

pub struct MatyasEnv(pub f32, pub f32);

impl Fitness for MatyasEnv {

    fn score(&self, candidate: &[f32]) -> f32 {
        let mut x = candidate[0];
        let mut y = candidate[1];
        x += self.0;
        y += self.1;
        -(0.26 * (x.powi(2) + y.powi(2)) - 0.48 * x * y)
    }

}

