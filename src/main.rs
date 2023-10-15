extern crate ndarray;
extern crate rand;

use rand::distributions::{Distribution, Uniform};
use ndarray::{Array, Array1, Array2};
use std::f64::consts::E;

// Rand util
fn rand_array1(len: usize, min: f64, max: f64) -> Array1<f64> {
        let dist = Uniform::from(min..max);
        let mut rng = rand::thread_rng();
        Array::from_shape_fn(len, |_| dist.sample(&mut rng))
}

fn rand_array2(shape: (usize, usize), min: f64, max: f64) -> Array2<f64> {
        let dist = Uniform::from(min..max);
        let mut rng = rand::thread_rng();
        Array::from_shape_fn(shape, |_| dist.sample(&mut rng))
}

// Traits
pub trait Layer<A> {
    fn forward(&self, input: Array1<A>) -> Array1<A>;
}

pub trait Activation<A> {
    fn apply(&self, input: Array1<A>) -> Array1<A>;
}

// Activation functions
pub struct Sigmoid;

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

impl Activation<f64> for Sigmoid {
    fn apply(&self, input: Array1<f64>) -> Array1<f64> {
        input.map(|&x| sigmoid(x))
    }
}

// Layers
pub struct DenseLayer<T, A: Activation<T>> {
    weights: Array2<T>,
    biases: Array1<T>,
    activation: A
}
impl<A: Activation<f64>> DenseLayer<f64, A> {
    pub fn new(input_size: usize, output_size: usize, activation: A) -> Self {
        let weights = rand_array2((input_size, output_size), 0.0, 1.0);
        let biases: Array1<f64> = Array1::<f64>::ones(output_size);

        DenseLayer {
            weights,
            biases,
            activation
        }
    }
}

impl <A: Activation<f64>> Layer<f64> for DenseLayer<f64, A> {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        self.activation.apply(input.dot(&self.weights) + &self.biases)
    }
}
pub struct Softmax;

impl Layer<f64> for Softmax {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        let max_val = input.sum();
        let exps = input.mapv(|x| E.powf(x - max_val));
        let sum_exps = exps.sum();
        exps / sum_exps
    }
}

// Model
pub struct Model<T> {
    layers: Vec<Box<dyn Layer<T>>>
}

impl<T> Model<T> {
    pub fn new() -> Self {
        Model {
            layers: Vec::new()
        }
    }

    pub fn add_layer<L: Layer<T> + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn forward(&self, mut data: Array1<T>) -> Array1<T> {
        for layer in &self.layers {
            data = layer.forward(data);
        }
        data
    }
}

fn main() {
    let input = rand_array1(3, -10.0, 10.0);

    let mut model = Model::new();
    model.add_layer(DenseLayer::new(3, 10, Sigmoid));
    model.add_layer(DenseLayer::new(10, 10, Sigmoid));
    model.add_layer(Softmax);

    let probs = model.forward(input);
    print!("{}\n", probs)
}
