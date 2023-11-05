extern crate ndarray;
extern crate rand;

use ndarray::{Array, Array1, Array2, Axis, s};
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
    fn forward(&mut self, input: Array1<f64>) -> Array1<f64>;
    fn backward(&self, d_output: Array1<f64>) -> (Array2<f64>, Array1<f64>);
    fn get_last_z(&self) -> Array1<f64>;
    fn update(&mut self, d_weights: Array2<f64>, learning_rate: f64);
}

pub trait Activation {
    fn apply(&self, input: &Array1<f64>) -> Array1<f64>;
    fn gradient(&self, y: &Array1<f64>) -> Array1<f64>;
}

// Activation functions
pub struct Sigmoid;

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

impl Activation for Sigmoid {
    fn apply(&self, input: &Array1<f64>) -> Array1<f64> {
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
    last_input: Array1<f64>,
    last_z: Array1<f64>
}
impl<A: Activation> DenseLayer<A> {
    pub fn new(input_size: usize, output_size: usize, activation: A) -> Self {
        let weights = rand_array2((input_size + 1, output_size), 0.0, 1.0);

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
        let delta = activation_prime * delta_output;
        let d_weights: Array2<f64> = self.last_input.clone().t().insert_axis(Axis(1)).dot(&delta.clone().insert_axis(Axis(0)));
        let weights_no_bias = self.weights.slice(s![1.., ..]); // Skip the bias row
        let d_input = weights_no_bias.dot(&delta);

        (d_weights, d_input)
    }
    fn get_last_z(&self) -> Array1<f64> {
        self.last_z.clone()
    }
    fn update(&mut self, d_weights: Array2<f64>, learning_rate: f64) {
        self.weights = &self.weights + learning_rate *  d_weights;
    }

}

pub struct Softmax {
    last_output: Array1<f64>
}
impl Softmax {
    fn new() -> Self {
        Softmax { last_output: Array1::<f64>::zeros(1) }
    }
}

impl Layer for Softmax {
    fn forward(&mut self, input: Array1<f64>) -> Array1<f64> {
        // Remove added first bias term
        let max_val = input.sum();
        let exps = input.mapv(|x| E.powf(x - max_val));
        let sum_exps = exps.sum();
        let probs = exps / sum_exps;
        self.last_output = probs.clone();
        probs
    }
    fn backward(&self, true_labels: Array1<f64>) -> (Array2<f64>, Array1<f64>) {
        // The gradient is the difference between the output probabilities and the true labels
        let d_z = &self.last_output - &true_labels;
        
        let d_weights = Array2::<f64>::zeros((0, 0));
        
        (d_weights, d_z)
    }
    fn get_last_z(&self) -> Array1<f64> {
        self.last_output.clone()
    }
    fn update(&mut self, _: Array2<f64>, _: f64) {}
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

    pub fn forward(&mut self, data: &Array1<f64>) -> Array1<f64> {
        let mut data = data.clone();
        for layer in &mut self.layers {
            data = layer.forward(data);
        }
        data
    }
    pub fn backward(&mut self, y: &Array1<f64>, learning_rate: f64) {

        let mut delta = y.clone();
        let mut d_weights;
        for layer in self.layers.iter_mut().rev() {
            (d_weights, delta) = layer.backward(delta);
            layer.update(d_weights, learning_rate);
        }
    }
    pub fn train(&mut self, x: &Array1<f64>, y: &Array1<f64>, epochs: i8, learning_rate: f64) {
        for epoch in 1..epochs {
            let probs = self.forward(x);
            let loss = cross_entropy_loss(&y, &probs);
            print!("Epoch {}, CE loss: {}\n", epoch, loss);
            print!("Prediction: {}\n", probs);
            self.backward(&y, learning_rate)
        }
    }

}

pub fn cross_entropy_loss(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
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
    model.add_layer(DenseLayer::new(3, 100, Sigmoid));
    model.add_layer(DenseLayer::new(100, 3, Sigmoid));
    model.add_layer(Softmax::new());

    model.train(&x, &y, 10, 0.1);
    // let probs = model.forward(&x);
    // print!("{}\n", probs);
    // let loss = cross_entropy_loss(&y, &probs);
    // print!("{}\n", loss);
    // model.backward(&y, 0.1);
    //
    // let probs = model.forward(&x);
    // print!("{}\n", probs);
}
