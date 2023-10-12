extern crate ndarray;
extern crate rand;

use rand::distributions::{Distribution, Uniform};
use ndarray::{Array, Array1, Array2};
use std::f64::consts::E;


fn rand_array2(shape: (usize, usize), min: f64, max: f64) -> Array2<f64> {
        let dist = Uniform::from(min..max);
        let mut rng = rand::thread_rng();
        Array::from_shape_fn(shape, |_| dist.sample(&mut rng))
}

trait Layer<A> {
    fn forward(&self, input: Array1<A>) -> Array1<A>;
}


pub struct DenseLayer {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = rand_array2((input_size, output_size), 0.0, 1.0);
        let biases: Array1<f64> = Array1::<f64>::ones(output_size);

        DenseLayer {
            weights,
            biases
        }
    }
}

impl Layer<f64> for DenseLayer {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        let result = input.dot(&self.weights);
        result + &self.biases
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

pub struct SoftmaxLayer;
pub struct Sigmoid;

impl Layer<f64> for SoftmaxLayer {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        let max_val = input.sum();
        let exps = input.mapv(|x| E.powf(x - max_val));
        let sum_exps = exps.sum();
        exps / sum_exps
    }
}

impl Layer<f64> for Sigmoid {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        input.map(|&x| sigmoid(x))
    }
}

fn main() {
    let input = Array1::<f64>::ones(3);

    let layer1 = DenseLayer::new(3, 10);
    let layer2 = DenseLayer::new(10, 10);
    let softmax = SoftmaxLayer;
    let sigmoid = Sigmoid;

    let act1 = layer1.forward(input);
    let act1s = sigmoid.forward(act1);
    let act2 = layer2.forward(act1s);
    let act2s = sigmoid.forward(act2);
    let probs = softmax.forward(act2s);
    print!("{}\n", probs)
}
