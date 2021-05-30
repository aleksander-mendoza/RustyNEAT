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

impl<X: Num> Neat<X> {
    pub fn get_random_activation_function(&mut self) -> fn(X) -> X {
        let r: f32 = rand::random();
        self.activations[self.activations.len() * r as usize]
    }
}

impl<X: Num> Neat<X> where Standard: Distribution<X> + Distribution<f64> {
    pub fn random_weight_generator(&self) -> X {
        rand::random()
    }

    pub fn new_cppn(&mut self) -> CPPN<X> {
        let (cppn, inno) = CPPN::new(self.input_size, self.output_size, self.get_global_innovation_no(), || self.random_weight_generator());
        self.set_global_innovation_no(inno);
        cppn
    }

    pub fn new_cppns(&mut self, num: usize) -> Vec<CPPN<X>> {
        let mut vec = Vec::with_capacity(num);
        let inno = self.get_global_innovation_no();
        let mut new_inno = 0;
        for _ in 0..num {
            // All the created CPPNs share the same innovation numbers
            // but only differ in randomly initialised weights
            let (cppn, updated_inno) = CPPN::new(self.input_size, self.output_size, inno, || self.random_weight_generator());
            new_inno = updated_inno;
            vec.push(cppn);
        }
        self.set_global_innovation_no(new_inno);
        vec
    }

    pub fn get_input_size(&self) -> usize {
        self.input_size
    }
    pub fn get_output_size(&self) -> usize {
        self.output_size
    }
    pub fn new_default(input_size: usize, output_size: usize) -> Self {
        Self::new(vec![activations::identity], input_size, output_size)
    }
    pub fn new(activations: Vec<fn(X) -> X>, input_size: usize, output_size: usize) -> Self {
        Self { global_innovation_no: 0, activations, input_size, output_size }
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
    /**returns true if successful*/
    pub fn add_connection_if_possible(&mut self, cppn: &mut CPPN<X>, from: usize, to: usize) -> bool {
        let inno = self.get_global_innovation_no();
        let w = self.random_weight_generator();
        let new_inno = cppn.add_connection_if_possible(from, to, w, inno);
        self.set_global_innovation_no(new_inno);
        new_inno != inno
    }
    /**Returns index of the newly created node*/
    pub fn add_node(&mut self, cppn: &mut CPPN<X>, edge_index: usize) {
        let inno = self.get_global_innovation_no();
        let af = self.get_random_activation_function();
        let new_inno = cppn.add_node(edge_index, af, inno);
        self.set_global_innovation_no(new_inno);
    }

    /**Randomly adds a new connection, but may fail if such change would result in recurrent
    neural net instead of feed-forward one (so acyclicity must be preserved). Returns true if successfully
    added a new edge*/
    pub fn attempt_to_add_random_connection(&mut self, cppn: &mut CPPN<X>) -> bool {
        self.add_connection_if_possible(cppn, cppn.get_random_node(), cppn.get_random_node())
    }

    pub fn add_random_node(&mut self, cppn: &mut CPPN<X>) {
        self.add_node(cppn, cppn.get_random_edge())
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