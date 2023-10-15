extern crate ndarray;
extern crate rand;

use ndarray::{Array, Array1, Array2, Axis};
use rand::distributions::{Distribution, Uniform};
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
pub trait Layer {
    fn forward(&self, input: Array1<f64>) -> Array1<f64>;
}

pub trait Activation {
    fn apply(&self, input: Array1<f64>) -> Array1<f64>;
    fn gradient(&self, y: &Array1<f64>) -> Array1<f64>;
}

// Activation functions
pub struct Sigmoid;

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

impl Activation for Sigmoid {
    fn apply(&self, input: Array1<f64>) -> Array1<f64> {
        input.map(|&x| sigmoid(x))
    }
    fn gradient(&self, y: &Array1<f64>) -> Array1<f64> {
        y * &(Array1::<f64>::ones(y.raw_dim()) - y)
    }
}

// Layers
pub struct DenseLayer<A: Activation> {
    weights: Array2<f64>,
    activation: A,
}
impl<A: Activation> DenseLayer<A> {
    pub fn new(input_size: usize, output_size: usize, activation: A) -> Self {
        let weights = rand_array2((input_size + 1, output_size), 0.0, 1.0);

        DenseLayer {
            weights,
            activation,
        }
    }
    pub fn backward(&self, d_output: Array1<f64>, z: Array1<f64>) -> (Array2<f64>, Array1<f64>) {
        let activation_prime = self.activation.gradient(&z);
        let d_z = d_output.dot(&self.weights.t()) * activation_prime;
        let d_weights: Array2<f64> = z.insert_axis(Axis(1)).dot(&d_output.insert_axis(Axis(0)));
        (d_weights, d_z)
    }
}

impl<A: Activation> Layer for DenseLayer<A> {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        // Add in the one at each layer
        let input = ndarray::concatenate(
            ndarray::Axis(0),
            &[ndarray::array![1.0].view(), input.view()],
        )
        .unwrap();
        self.activation.apply(input.dot(&self.weights))
    }
}
pub struct Softmax;

impl Layer for Softmax {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        // Remove added first bias term
        let max_val = input.sum();
        let exps = input.mapv(|x| E.powf(x - max_val));
        let sum_exps = exps.sum();
        exps / sum_exps
    }
}

// Model
pub struct Model {
    layers: Vec<Box<dyn Layer>>,
}

impl Model {
    pub fn new() -> Self {
        Model { layers: Vec::new() }
    }

    pub fn add_layer<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn forward(&self, mut data: Array1<f64>) -> Array1<f64> {

        for layer in &self.layers {
            data = layer.forward(data);
        }
        data
    }
}

pub fn cross_entropy_loss(y_true: Array1<f64>, y_pred: Array1<f64>) -> f64 {
    let all_losses =
        y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y, &p)| if y == 0.0 { 0.0 } else { y * p.log(E) });
    let negative_loss: f64 = all_losses.sum();
    negative_loss * -1.0
}

fn main() {
    let x = rand_array1(3, -10.0, 10.0);
    let y = ndarray::array![0.0, 1.0, 0.0];

    let mut model = Model::new();
    model.add_layer(DenseLayer::new(3, 10, Sigmoid));
    model.add_layer(DenseLayer::new(10, 3, Sigmoid));
    model.add_layer(Softmax);

    let probs = model.forward(x);
    print!("{}\n", probs);
    let loss = cross_entropy_loss(y, probs);
    print!("{}\n", loss);
}
