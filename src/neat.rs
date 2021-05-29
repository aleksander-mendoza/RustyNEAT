use rand::Rng;
use rand::rngs::StdRng;
use crate::num::Num;
use crate::activations;
use crate::cppn::CPPN;
use rand::distributions::{Standard, Distribution};

pub struct Neat<X: Num> {
    global_innovation_no: usize,
    activations: Vec<fn(X) -> X>,
    input_size: usize,
    output_size: usize,
}

impl<X: Num> Neat<X> where Standard: Distribution<X>{
    pub fn get_input_size(&self) -> usize{
        self.input_size
    }
    pub fn get_output_size(&self) -> usize{
        self.output_size
    }
    pub fn new_default(input_size: usize, output_size: usize) -> Self {
        Self::new(vec![activations::identity],input_size,output_size)
    }
    pub fn new(activations: Vec<fn(X) -> X>, input_size: usize, output_size: usize) -> Self {
        Self { global_innovation_no: 0, activations, input_size, output_size }
    }

    pub fn random_weight_generator(&self) -> X {
        rand::random()
    }

    pub fn set_global_innovation_no(&mut self, val: usize) {
        self.global_innovation_no = val;
    }

    pub fn get_global_innovation_no(&self) -> usize {
        self.global_innovation_no
    }

    pub fn activation_functions_len(&self) -> usize {
        self.activations.len()
    }

    pub fn get_activation_function(&self, i: usize) -> fn(X) -> X {
        self.activations[i]
    }

    pub fn new_cppn(&mut self) -> CPPN<X> {
        let (cppn, inno) = CPPN::new(self.input_size,self.output_size, self.get_global_innovation_no(), || self.random_weight_generator());
        self.set_global_innovation_no(inno);
        cppn
    }

    pub fn get_input_slice_mut<'a>(&self, input_buffer: &'a mut [X]) -> &'a mut [X] {
        &mut input_buffer[..self.input_size]
    }

    pub fn get_input_slice<'a>(&self, input_buffer: &'a [X]) -> &'a [X] {
        &input_buffer[..self.input_size]
    }

    pub fn get_output_slice_mut<'a>(&self, input_buffer: &'a mut [X]) -> &'a mut [X] {
        &mut input_buffer[self.input_size..self.input_size + self.output_size]
    }

    pub fn get_output_slice<'a>(&self, input_buffer: &'a [X]) -> &'a [X] {
        &input_buffer[self.input_size..self.input_size + self.output_size]
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cppn_1() {
        let mut neat = Neat::<f64>::new_default(3, 4);
        let cppn = neat.new_cppn();
        let net = cppn.build_feed_forward_net();
        let mut buff = net.new_input_buffer();
        net.run(buff.as_mut_slice());
        neat.get_output_slice(buff.as_slice());
    }
}