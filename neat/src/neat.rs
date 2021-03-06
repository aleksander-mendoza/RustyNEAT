use rand::Rng;
use rand::rngs::StdRng;
use crate::num::Num;
use crate::activations;
use crate::cppn::CPPN;
use rand::distributions::{Standard, Distribution};
use crate::activations::{ActFn, ALL_ACT_FN};
use std::iter::FromIterator;
use crate::util::RandRange;

pub struct Neat {
    global_innovation_no: usize,
    activations: Vec<&'static ActFn>,
    input_size: usize,
    output_size: usize,
}

impl Neat {
    pub fn get_activation_functions(&self) -> &Vec<&'static ActFn> {
        &self.activations
    }

    pub fn get_random_activation_function(&self) -> &'static ActFn {
        self.activations[self.activations.len().random()]
    }

    pub fn new_cppn<X: Num>(&mut self) -> CPPN<X> {
        let (cppn, inno) = CPPN::new(self.input_size, self.output_size, self.get_global_innovation_no());
        self.set_global_innovation_no(inno);
        cppn
    }

    pub fn new_cppns<X: Num>(&mut self, num: usize) -> Vec<CPPN<X>> {
        let mut vec = Vec::with_capacity(num);
        if num==0{return vec;}
        let inno = self.get_global_innovation_no();
        let (cppn, new_inno) = CPPN::new(self.input_size, self.output_size, inno);
        vec.push(cppn);
        for _ in 1..num {
            // All the created CPPNs share the same innovation numbers
            // but only differ in randomly initialised weights
            let (cppn, updated_inno) = CPPN::new(self.input_size, self.output_size, inno);
            assert_eq!(new_inno, updated_inno);
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
        Self::new(Vec::from_iter(ALL_ACT_FN.iter()), input_size, output_size)
    }
    pub fn new(activations: Vec<&'static ActFn>, input_size: usize, output_size: usize) -> Self {
        Self { global_innovation_no: 0, activations, input_size, output_size }
    }

    pub fn set_global_innovation_no(&mut self, val: usize) {
        assert!(val>=self.global_innovation_no,"Was {} and new value is {}",self.global_innovation_no, val);
        self.global_innovation_no = val;
    }

    pub fn get_global_innovation_no(&self) -> usize {
        self.global_innovation_no
    }

    pub fn activation_functions_len(&self) -> usize {
        self.activations.len()
    }

    pub fn get_activation_function(&self, i: usize) -> &'static ActFn {
        self.activations[i]
    }

    pub fn get_input_slice_mut<'a, X>(&self, input_buffer: &'a mut [X]) -> &'a mut [X] {
        &mut input_buffer[..self.input_size]
    }

    pub fn get_input_slice<'a, X>(&self, input_buffer: &'a [X]) -> &'a [X] {
        &input_buffer[..self.input_size]
    }

    pub fn get_output_slice_mut<'a, X>(&self, input_buffer: &'a mut [X]) -> &'a mut [X] {
        &mut input_buffer[self.input_size..self.input_size + self.output_size]
    }

    pub fn get_output_slice<'a, X>(&self, input_buffer: &'a [X]) -> &'a [X] {
        &input_buffer[self.input_size..self.input_size + self.output_size]
    }
    /**returns true if successful*/
    pub fn add_connection_if_possible<X: Num>(&mut self, cppn: &mut CPPN<X>, from: usize, to: usize) -> bool {
        let inno = self.get_global_innovation_no();
        let new_inno = cppn.add_connection_if_possible(from, to, X::random(), inno);
        self.set_global_innovation_no(new_inno);
        new_inno != inno
    }
    /**Returns index of the newly created node*/
    pub fn add_node<X: Num>(&mut self, cppn: &mut CPPN<X>, edge_index: usize) {
        let inno = self.get_global_innovation_no();
        let af = self.get_random_activation_function();
        let new_inno = cppn.add_node(edge_index, af, inno);
        assert!(inno<new_inno,"Was {} and updated to {}",inno,new_inno);
        self.set_global_innovation_no(new_inno);
    }

    /**Randomly adds a new connection, but may fail if such change would result in recurrent
    neural net instead of feed-forward one (so acyclicity must be preserved). Returns true if successfully
    added a new edge*/
    pub fn add_random_connection<X: Num>(&mut self, cppn: &mut CPPN<X>) -> bool {
        assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        let b = self.add_connection_if_possible(cppn, cppn.get_random_node(), cppn.get_random_node());
        assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        b
    }

    pub fn add_random_node<X: Num>(&mut self, cppn: &mut CPPN<X>) {
        assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        self.add_node(cppn, cppn.get_random_edge());
        assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
    }

    pub fn make_output_buffer<'b, 'x, X: Num + 'x, I: Iterator<Item=&'x CPPN<X>>>(&'b self, population: I) -> Option<Vec<X>> {
        population.map(CPPN::node_count).max().map(|m| vec![X::zero(); m])
    }

    pub fn mutate_population<'b, 'x, X: Num + 'x, I: Iterator<Item=&'x mut CPPN<X>>>(&'b mut self, population: I,
                                                                                     node_insertion_prob: f32,
                                                                                     edge_insertion_prob: f32,
                                                                                     activation_fn_mutation_prob: f32,
                                                                                     weight_mutation_prob: f32,
                                                                                     enable_edge_prob: f32,
                                                                                     disable_edge_prob: f32) {
        for cppn in population {
            self.mutate(cppn,node_insertion_prob,
                        edge_insertion_prob,
                        activation_fn_mutation_prob,
                        weight_mutation_prob,
                        enable_edge_prob,
                        disable_edge_prob)
        }
    }
    pub fn mutate<X: Num>(&mut self, cppn: &mut CPPN<X>,
                          node_insertion_prob: f32,
                          edge_insertion_prob: f32,
                          activation_fn_mutation_prob: f32,
                          weight_mutation_prob: f32,
                          enable_edge_prob: f32,
                          disable_edge_prob: f32) {
        let was_acyclic = cppn.is_acyclic();
        assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        if f32::random() < node_insertion_prob {
            self.add_random_node(cppn)
        }
        assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        cppn.assert_invariants("after add random node");
        if f32::random() < edge_insertion_prob {
            self.add_random_connection(cppn);
        }
        assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        cppn.assert_invariants("after add random connection");
        for edge_index in 0..cppn.edge_count() {
            if f32::random() < weight_mutation_prob {
                cppn.set_weight(edge_index, X::random())
            }
            if cppn.is_enabled(edge_index){
                if f32::random() < disable_edge_prob {
                    cppn.set_enabled(edge_index, false);
                }
            }else{
                if f32::random() < enable_edge_prob {
                    cppn.set_enabled(edge_index, true);
                }
            }
        }
        assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        for node_index in 0..cppn.node_count() {
            if f32::random() < activation_fn_mutation_prob {
                cppn.set_activation(node_index, self.get_random_activation_function());
            }
        }
        assert!(cppn.edges().all(|e| e.innovation_no() <= self.get_global_innovation_no()));
        cppn.assert_invariants("after mutate");
        assert_eq!(was_acyclic, cppn.is_acyclic());
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{FeedForwardNetOpenCL, FeedForwardNetPicbreeder};
    use rand::random;
    use ocl::core::default_platform;
    use ndalgebra::mat::{Mat, MatError};
    use ndalgebra::context::Context;
    use ndalgebra::lin_alg_program::LinAlgProgram;

    #[test]
    fn cppn_1() {
        let mut neat = Neat::new_default(3, 4);
        let cppn = neat.new_cppn::<f32>();
        let net = cppn.build_feed_forward_net();
        let mut out = [0f32, 0.0, 0.0, 0.0];
        net.run(&[4f32, 5.0, 5.0], &mut out);
    }

    #[test]
    fn cppn_2() {
        let mut neat = Neat::new_default(2, 1);
        let cppn = neat.new_cppn::<f32>();
        let net = cppn.build_feed_forward_net();
        let mut out = [0f32];
        net.run(&[1f32, 2.0], &mut out);
    }

    #[test]
    fn cppn_3() {
        let mut neat = Neat::new_default(2, 1);
        let mut cppns = neat.new_cppns::<f64>(16);
        neat.mutate_population(cppns.iter_mut(), 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
        let crossed_over = cppns[0].crossover(&cppns[1]);
    }

    #[test]
    fn cppn_4() -> Result<(), MatError> {
        let mut neat = Neat::new_default(3, 4);
        let cppn = neat.new_cppn::<f32>();
        let net = cppn.build_feed_forward_net();
        let mut out = [0f32, 0.0, 0.0, 0.0];
        let input = [4f32, 5.0, 5.0];
        net.run(&input, &mut out);
        println!("{}", net);
        let p = LinAlgProgram::default()?;
        let gpu = FeedForwardNetOpenCL::new(&p,&net)?;
        let i = Mat::from_slice_infer_wildcard(&p,&input,&[1, -1])?;
        let out3 = gpu.run(&i)?;
        assert_eq!(out, out3.to_vec()?.as_slice());

        Ok(())
    }

    #[test]
    fn cppn_5() -> Result<(), MatError> {
        let mut neat = Neat::new_default(3, 4);
        let cppn = neat.new_cppn::<f32>();
        let net = cppn.build_feed_forward_net();
        let mut out = [0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let input = [0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        net.run(&input[0..3], &mut out[0..4]);
        net.run(&input[3..6], &mut out[4..8]);

        println!("{}", net);
        let p = LinAlgProgram::default()?;
        let input = Mat::array1(&p,input)?.reshape2(2,3)?;
        let gpu = FeedForwardNetOpenCL::new(&p, &net)?;
        let out3 = gpu.run(&input)?;
        assert_eq!(out, out3.to_vec()?.as_slice());

        Ok(())
    }


    #[test]
    fn cppn_6() -> Result<(), MatError> {
        let mut neat = Neat::new_default(2, 1);
        let cppn = neat.new_cppn::<f32>();
        let net = cppn.build_feed_forward_net();
        let mut out = [0f32, 0., 0., 0.];
        let dimensions = [2usize, 2];
        let pixel_sizes = [1f32, 1.0];
        let pixel_offsets = [0f32, 0.];
        net.run(&[0f32, 0.0], &mut out[0..1]);
        net.run(&[1f32, 0.0], &mut out[1..2]);
        net.run(&[0f32, 1.0], &mut out[2..3]);
        net.run(&[1f32, 1.0], &mut out[3..4]);
        println!("{}", net);
        println!("{}", net.picbreeder_view(None, false).unwrap());
        let p = LinAlgProgram::default()?;
        let gpu = FeedForwardNetPicbreeder::new(&p, &net, None, false)?;
        let out2 = gpu.run(&dimensions, &pixel_sizes, &pixel_offsets)?;
        assert_eq!(out2.shape(), &[2,2,1]);
        assert_eq!(out, out2.to_vec()?.as_slice());
        Ok(())
    }

    #[test]
    fn cppn_7() -> Result<(), MatError> {
        let mut neat = Neat::new_default(3, 2);
        let cppn = neat.new_cppn::<f32>();
        let net = cppn.build_feed_forward_net();
        let mut out = [0f32, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.];
        let dimensions = [2usize, 2, 2];
        let pixel_sizes = [rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>()];
        let pixel_offsets = [rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>()];
        net.run(&[0f32 * pixel_sizes[0] + pixel_offsets[0], 0.0 * pixel_sizes[1] + pixel_offsets[1], 0. * pixel_sizes[2] + pixel_offsets[2]], &mut out[0..2]);
        net.run(&[1f32 * pixel_sizes[0] + pixel_offsets[0], 0.0 * pixel_sizes[1] + pixel_offsets[1], 0. * pixel_sizes[2] + pixel_offsets[2]], &mut out[2..4]);
        net.run(&[0f32 * pixel_sizes[0] + pixel_offsets[0], 1.0 * pixel_sizes[1] + pixel_offsets[1], 0. * pixel_sizes[2] + pixel_offsets[2]], &mut out[4..6]);
        net.run(&[1f32 * pixel_sizes[0] + pixel_offsets[0], 1.0 * pixel_sizes[1] + pixel_offsets[1], 0. * pixel_sizes[2] + pixel_offsets[2]], &mut out[6..8]);
        net.run(&[0f32 * pixel_sizes[0] + pixel_offsets[0], 0.0 * pixel_sizes[1] + pixel_offsets[1], 1. * pixel_sizes[2] + pixel_offsets[2]], &mut out[8..10]);
        net.run(&[1f32 * pixel_sizes[0] + pixel_offsets[0], 0.0 * pixel_sizes[1] + pixel_offsets[1], 1. * pixel_sizes[2] + pixel_offsets[2]], &mut out[10..12]);
        net.run(&[0f32 * pixel_sizes[0] + pixel_offsets[0], 1.0 * pixel_sizes[1] + pixel_offsets[1], 1. * pixel_sizes[2] + pixel_offsets[2]], &mut out[12..14]);
        net.run(&[1f32 * pixel_sizes[0] + pixel_offsets[0], 1.0 * pixel_sizes[1] + pixel_offsets[1], 1. * pixel_sizes[2] + pixel_offsets[2]], &mut out[14..16]);
        println!("{}", net);
        println!("{}", net.picbreeder_view(None,false).unwrap());
        let p = LinAlgProgram::default()?;
        let gpu = FeedForwardNetPicbreeder::new(&p, &net, None, false)?;
        let out2 = gpu.run(&dimensions, &pixel_sizes, &pixel_offsets)?;
        assert_eq!(out2.shape(), &[2,2,2,2]);
        assert_eq!(out, out2.to_vec()?.as_slice());
        Ok(())
    }

    #[test]
    fn cppn_8()-> Result<(), MatError> {
        let mut neat = Neat::new_default(3, 3);
        let mut cppns = neat.new_cppns::<f32>(16);
        for _ in 0..10 {
            for cppn in cppns.iter(){
                let net = cppn.build_feed_forward_net();
                let in_vec = [rand::random::<f32>(),rand::random(),rand::random()];
                let mut out_vec = [0f32,0.,0.];
                net.run(&in_vec, &mut out_vec);
            }
            neat.mutate_population(cppns.iter_mut(), 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);

        }
        let crossed_over = cppns[0].crossover(&cppns[1]);
        Ok(())
    }

    #[test]
    fn cppn_9() -> Result<(), MatError>{
        let mut neat = Neat::new_default(4, 1);
        let mut cppn = neat.new_cppn::<f32>();
        for _ in 0..16 {
            neat.mutate(&mut cppn, 0.1, 0.2, 0.1, 0.1, 0.1, 0.01);
        }
        let net = cppn.build_feed_forward_net();
        let p = LinAlgProgram::default()?;
        println!("{}",net.substrate_view(2,2).unwrap());
        let gpu_net = net.to_substrate(2,Some(2),&p)?;
        let in_neurons = [1f32,2.,3.,4.,5.,6.];
        let out_neurons = [0.1f32,0.2,0.3,0.4,0.5,0.6];
        let ann = gpu_net.run(
            &Mat::from_slice(&p,&in_neurons,&[3,2])?,
            &Mat::from_slice(&p,&out_neurons,&[3,2])?)?;
        let mut ann2 = [0f32;3*3];
        for (row, i) in in_neurons.chunks(2).enumerate(){
            for (col, o) in out_neurons.chunks(2).enumerate(){
                let input = [i[0],i[1],o[0],o[1]];
                let idx = col*3+row;
                net.run(&input,&mut ann2[idx..idx+1]);
            }
        }
        assert_eq!(ann2,ann.to_vec()?.as_slice());
        Ok(())
    }
}