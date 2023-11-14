extern crate ndarray;
extern crate rand;

use ndarray::{Array, Array1, Array2, Array3, Axis, s};
use rand_distr::{Uniform, Distribution};
use std::f64::consts::E;
use std::iter::zip;

// Rand util
fn rand_array1(len: usize, min: f64, max: f64) -> Array1<f64> {
    let dist = Uniform::from(min..max);
    let mut rng = rand::thread_rng();
    Array::from_shape_fn(len, |_| dist.sample(&mut rng))
}

// Util
fn argmax_1d(array: &Array1<f64>) -> usize {
    array.iter()
         .enumerate()
         .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
         .map(|(index, _)| index)
         .unwrap_or(0)
}

// Traits

// Neural Network Layer
pub trait Layer {
    fn forward(&mut self, input: Array1<f64>) -> Array1<f64>;
    fn backward(&self, d_output: Array1<f64>) -> (Array2<f64>, Array1<f64>);
    fn update(&mut self, d_weights: Array2<f64>, learning_rate: f64);
    fn get_weights(&self) -> &Array2<f64>;
}

// Activation Function
pub trait Activation {
    fn apply(&self, input: &Array1<f64>) -> Array1<f64>;
    fn gradient(&self, z: &Array1<f64>) -> Array1<f64>;
    fn get_default_weight_init(&self, shape: (usize, usize)) -> Array2<f64>;
}

// Activation functions definitions
// Based on standard inits per activation function in: https://pytorch.org/docs/stable/nn.init.html

pub struct Linear;

impl Activation for Linear {
    fn apply(&self, input: &Array1<f64>) -> Array1<f64> {
        input.clone()
    }
    fn gradient(&self, z: &Array1<f64>) -> Array1<f64> {
        Array1::<f64>::ones(z.raw_dim())
    }
    fn get_default_weight_init(&self, shape: (usize, usize)) -> Array2<f64> {
        Array2::<f64>::ones(shape)
    }
}

pub struct Relu;

impl Activation for Relu {
    fn apply(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|x| x.max(0.0))
    }
    fn gradient(&self, z: &Array1<f64>) -> Array1<f64> {
        z.mapv(|x| if x > 0.0 {1.0} else {0.0})
    }
    fn get_default_weight_init(&self, shape: (usize, usize)) -> Array2<f64> {
        Array2::<f64>::from_elem(shape, (2.0_f64).sqrt())
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn apply(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|x| sigmoid(x))
    }
    fn gradient(&self, z: &Array1<f64>) -> Array1<f64> {
        z * &(Array1::<f64>::ones(z.raw_dim()) - z)
    }
    fn get_default_weight_init(&self, shape: (usize, usize)) -> Array2<f64> {
        Array2::<f64>::ones(shape)
    }
}

// Layers
pub struct DenseLayer<A: Activation> {
    weights: Array2<f64>,
    activation: A,
    last_input: Array1<f64>,
    last_z: Array1<f64>
}
impl<A: Activation> DenseLayer<A> {
    pub fn new(input_size: usize, output_size: usize, activation: A) -> Self {
        let weights = activation.get_default_weight_init((input_size + 1, output_size)); 

        DenseLayer {
            weights,
            activation,
            last_input: Array::zeros(input_size),
            last_z: Array::zeros(output_size)
        }
    }
}

impl<A: Activation> Layer for DenseLayer<A> {
    fn forward(&mut self, input: Array1<f64>) -> Array1<f64> {
        // Add in the one at each layer
        let input = ndarray::concatenate(
            ndarray::Axis(0),
            &[ndarray::array![1.0].view(), input.view()],
        )
        .unwrap();
        let z = input.dot(&self.weights);
        let output = self.activation.apply(&z);

        self.last_input = input.clone();
        self.last_z = z;

        output
    }
    fn backward(&self, delta_output: Array1<f64>) -> (Array2<f64>, Array1<f64>) {
        let activation_prime = self.activation.gradient(&self.last_z);
        let delta = &activation_prime * &delta_output;
        let d_weights: Array2<f64> = self.last_input.clone().t().insert_axis(Axis(1)).dot(&delta.clone().insert_axis(Axis(0)));
        let weights_no_bias = self.weights.slice(s![1.., ..]);
        let d_input = weights_no_bias.dot(&delta);

        (d_weights, d_input)
    }
    fn update(&mut self, d_weights: Array2<f64>, learning_rate: f64) {
        let update = learning_rate * d_weights;
        self.weights = &self.weights + -1.0 * update;
    }
    fn get_weights(&self) -> &Array2<f64> {
        &self.weights
    }
}

pub struct Softmax;

impl Activation for Softmax {
    fn apply(&self, input: &Array1<f64>) -> Array1<f64> {
        // Remove added first bias term
        let max_val = input.sum();
        let exps = input.mapv(|x| E.powf(x - max_val));
        let sum_exps = exps.sum();
        let probs = exps / sum_exps;
        probs
    }
    fn gradient(&self, z: &Array1<f64>) -> Array1<f64> {
        Array1::ones(z.shape()[0])
    }
    fn get_default_weight_init(&self, shape: (usize, usize)) -> Array2<f64> {
        Array2::<f64>::ones(shape)
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

    fn update(&mut self, weights_per_layer: Vec<Array2<f64>>, learning_rate: f64) {
        for (layer, d_weights) in zip(self.layers.iter_mut(), weights_per_layer) {
            layer.update(d_weights, learning_rate);
        }
    }
    pub fn add_layer<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn forward(&mut self, data: &Array1<f64>) -> Array1<f64> {
        let mut data = data.clone();
        for layer in &mut self.layers {
            data = layer.forward(data);
        }
        if data.iter().any(|&x| x.is_nan()) {
            panic!("Numeric instability error!")
        }
        data
    }
    pub fn backward(&self, error: &Array1<f64>) -> Vec<Array2<f64>> {
        let mut delta = error.clone();
        let mut d_weights;
        let mut all_d_weights: Vec<Array2<f64>> = Vec::new();
        for layer in self.layers.iter().rev() {
            (d_weights, delta) = layer.backward(delta);
            all_d_weights.push(d_weights.to_owned())
        }
        all_d_weights.reverse();
        all_d_weights
    }
    pub fn train(&mut self, x: &Array2<f64>, y: &Array2<f64>, epochs: u16, learning_rate: f64) {
        // Devide x and y into batches by batch size

        let batch_size = x.len_of(Axis(0));

        for epoch in 0..epochs {
            let mut batch_d_weights_sum: Vec<Array2<f64>> = Vec::new();
            let mut initialized = false;

            print!("Epoch: {epoch}\n");
            for (x_i, y_i) in zip(x.axis_iter(Axis(0)), y.axis_iter(Axis(0))) {
                let probs = self.forward(&x_i.to_owned());
                let error = probs - &y_i;
                let all_d_weights = self.backward(&error);

                let probs = self.forward(&x_i.to_owned());
                let loss = cross_entropy_loss(&y_i.to_owned(), &probs);
                print!("Loss: {loss}\n");

                if !initialized {
                    batch_d_weights_sum = all_d_weights.iter().map(|dw| dw.clone()).collect();
                    initialized = true;
                } else {
                    for (sum, dw) in batch_d_weights_sum.iter_mut().zip(all_d_weights.iter()) {
                        *sum += dw;
                    }
                }
            }

            // Average the gradients across all mini-batches
            for layer_d_weights in batch_d_weights_sum.iter_mut() {
                *layer_d_weights /= batch_size as f64;
            }
            self.update(batch_d_weights_sum, learning_rate);


        }
    }
}

pub fn cross_entropy_loss(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let all_losses =
        y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y, &p)| if p == 0.0 { 0.0 } else { -y * p.log(E) });
    all_losses.sum()
}

fn main() {
    // let x = rand_array1(3, -0.5, 0.5);
    let x = ndarray::array![
        [10.0, 3.0, 1.1],
        [-10.0, 10.0, 10.0]
    ];
    let y = ndarray::array![
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ];

    let mut model = Model::new();
    model.add_layer(DenseLayer::new(3, 10, Sigmoid));
    model.add_layer(DenseLayer::new(10, 3, Softmax));

    model.train(&x, &y, 100, 0.01);
    // let probs = model.forward(&x);
    // print!("{}\n", probs);
    // let loss = cross_entropy_loss(&y, &probs);
    // print!("{}\n", loss);
    // model.backward(&y, 0.1);
    //
    // let probs = model.forward(&x);
    // print!("{}\n", probs);
}
