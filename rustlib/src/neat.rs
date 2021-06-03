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

impl<'x, X: 'x + Num> Neat<X> where Standard: Distribution<X> + Distribution<f64> {
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

    pub fn make_output_buffer<'b, I:Iterator<Item=&'b CPPN<X>>>(&'b self, population: I)->Option<Vec<X>>{
        population.map(CPPN::node_count).max().map(|m|vec![X::zero(); m])
    }

    pub fn mutate_population<'b, I:Iterator<Item=&'x mut CPPN<X>>>(&'b mut self, population: I,
                             node_insertion_prob: f32,
                             edge_insertion_prob: f32,
                             activation_fn_mutation_prob: f32,
                             weight_mutation_prob: f32) {
        for cppn in population {
            let was_acyclic = cppn.is_acyclic();
            if rand::random::<f32>() < node_insertion_prob {
                self.add_random_node(cppn)
            }
            if rand::random::<f32>() < edge_insertion_prob {
                self.attempt_to_add_random_connection(cppn);
            }
            for edge_index in 0..cppn.edge_count(){
                if rand::random::<f32>() < weight_mutation_prob {
                    cppn.set_weight(edge_index, self.random_weight_generator())
                }
            }
            for node_index in 0..cppn.node_count(){
                if rand::random::<f32>() < activation_fn_mutation_prob {
                    cppn.set_activation(node_index,self.get_random_activation_function())
                }
            }
            cppn.assert_invariants();
            assert_eq!(was_acyclic, cppn.is_acyclic());
        }
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
        let mut buff = net.make_output_buffer();
        net.run(buff.as_mut_slice());
        neat.get_output_slice(buff.as_slice());
    }

    #[test]
    fn cppn_2() {
        let mut neat = Neat::<f64>::new_default(2, 1);
        let cppn = neat.new_cppn();
        let net = cppn.build_feed_forward_net();
        let mut buff = net.make_output_buffer();
        net.run(buff.as_mut_slice());
        neat.get_output_slice(buff.as_slice());
    }

    #[test]
    fn cppn_3() {
        let mut neat = Neat::<f64>::new_default(2, 1);
        let mut cppns = neat.new_cppns(16);
        neat.mutate_population( cppns.iter_mut(),0.1,0.1,0.1,0.1);
        let crossed_over = cppns[0].crossover(&cppns[1]);
    }
}