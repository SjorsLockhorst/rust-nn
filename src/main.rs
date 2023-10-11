use ndarray::{Array, Ix1};

extern crate ndarray;

trait Layer<D> {
    fn forward(&self, input: Array<f64, D>) -> Array<f64, D>;
}
pub struct InputLayer {
    input_size: usize
}

impl Layer<Ix1> for InputLayer {
    fn forward(&self, input: Array<f64, Ix1>) -> Array<f64, Ix1> {
        // Placeholder implementation; you'd have the real logic here.
        input
    }
}



pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

pub struct SoftmaxLayer {
    size: usize
}




fn main() {
    print!("Hello world!\n")
}
