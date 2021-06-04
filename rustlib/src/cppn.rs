use crate::num::Num;
use crate::util::{Initializer, RandRange};
use rand::Rng;
use crate::activations;
use std::fmt::{Display, Formatter};
use std::num::NonZeroUsize;
use crate::activations::{ActFn};


pub struct FeedForwardNetOpenCL{

}

enum EdgeOrNode<X> {
    Node(usize, &'static ActFn),
    Edge(usize, X, usize),
}

pub struct FeedForwardNet<X: Num> {
    net: Vec<EdgeOrNode<X>>,
    len: usize,
    input_size: usize,
    output_size: usize,
}

impl<X: Num> FeedForwardNet<X> {
    pub fn get_input_size(&self) -> usize {
        self.input_size
    }
    pub fn get_output_size(&self) -> usize {
        self.output_size
    }
    pub fn run(&self, input_buffer: &[X], output_buffer: &mut [X]) {
        assert!(input_buffer.len() == self.get_input_size());
        let inout_size = self.input_size + self.output_size;
        let mut intermediate_buffer = vec![X::zero(); self.len - inout_size];
        for instruction in &self.net {
            match *instruction {
                EdgeOrNode::Edge(from, w, to) => {
                    let in_val = if from < self.input_size{
                        input_buffer[from]
                    }else if from < inout_size{
                        output_buffer[from-self.input_size]
                    } else{
                        intermediate_buffer[from-inout_size]
                    };
                    if to < inout_size{
                        assert!(to>=self.input_size);
                        output_buffer[to-self.input_size] += w * in_val;
                    }else{
                        intermediate_buffer[to-inout_size] += w * in_val;
                    }
                },
                EdgeOrNode::Node(from, activation) =>{
                    assert!(from>=self.input_size);
                    if from < inout_size{
                        let from = from - self.input_size;
                        output_buffer[from] = X::act_fn(activation)(output_buffer[from]);
                    }else {
                        let from = from - inout_size;
                        intermediate_buffer[from] = X::act_fn(activation)(intermediate_buffer[from]);
                    }
                }
            }
        }
    }
    pub fn compile_for_opencl(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "__kernel void feedforward(__global float * in,__global float * out,");
        writeln!(f, "                          size_t in_row_stride, size_t in_col_stride, ");
        writeln!(f, "                          size_t out_row_stride, size_t out_col_stride) {{");
        let mut was_written_to = vec![false; self.len];
        let inout_size = self.input_size + self.output_size;
        for instruction in &self.net {
            write!(f, "   ");
            match instruction {
                &EdgeOrNode::Node(idx, act_fn) => {
                    assert!(idx>self.input_size);
                    if idx < inout_size{
                        writeln!(f, "out[get_global_id(0)*out_row_stride+{}*out_col_stride] = {}(out[get_global_id(0)*out_row_stride+{}*out_col_stride]);", idx, act_fn.opencl_name(), idx)
                    }else{
                        writeln!(f, "register{} = {}(register{});", idx, act_fn.opencl_name(), idx)
                    }
                },
                &EdgeOrNode::Edge(from, weight, to) => {
                    assert!(to >= self.input_size); //cannot write to input node
                    if to < inout_size{
                        assert!(!was_written_to[to]); // this does not need to change
                        write!(f, "out[get_global_id(0)*out_row_stride+{}*out_col_stride] += ",to);
                    }else if was_written_to[to] { // variable already declared before
                        write!(f, "register{} += ", to);
                    } else { // first time write means that variable declaration is necessary
                        was_written_to[to] = true;
                        write!(f, "float register{} = ", to);
                    }
                    if from < self.input_size{
                        assert!(!was_written_to[from]); // input registers are never written to
                        write!(f, "in[get_global_id(0)*in_row_stride+{}*in_col_stride]",from);
                    }else if from < inout_size{
                        assert!(!was_written_to[from]); // this does not need to change
                        write!(f, "out[get_global_id(0)*out_row_stride+{}*out_col_stride]",from);
                    }else{
                        assert!(was_written_to[from]);
                        write!(f, "register{}", from);
                    }
                    writeln!(f, " * {};", weight)
                }
            };
        }
        write!(f, "}}");
        Ok(())
    }

}

impl<X: Num> Display for FeedForwardNet<X> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.compile_for_opencl(f)
    }
}

#[derive(Clone)]
struct Node {
    /**Initial nodes do not have any activation*/
    activation: Option<&'static ActFn>,
}

#[derive(Clone)]
pub struct CPPN<X: Num> {
    nodes: Vec<Node>,
    edges: Vec<Edge<X>>,
    input_size: usize,
    output_size: usize,
}

#[derive(Clone)]
struct Edge<X: Num> {
    innovation_no: usize,
    enabled: bool,
    from: usize,
    weight: X,
    to: usize,
}


impl<X: Num> CPPN<X> {
    fn build_edge_lookup_table(&self) -> Vec<Vec<(usize, usize)>> {
        let mut edge_lookup = Vec::initialize(self.nodes.len(), |_| Vec::new());
        for (edge_idx, edge) in self.edges.iter().enumerate() {
            let mut entry = &mut edge_lookup[edge.from];
            // assert!(!entry.contains(&edge.to));
            entry.push((edge.to, edge_idx));
        }
        edge_lookup
    }

    fn build_enabled_edge_only_lookup_table(&self) -> Vec<Vec<(usize, usize)>> {
        let mut edge_lookup = Vec::initialize(self.nodes.len(), |_| Vec::new());
        for (edge_idx, edge) in self.edges.iter().enumerate() {
            let mut entry = &mut edge_lookup[edge.from];
            // assert!(!entry.contains(&edge.to));
            if edge.enabled {
                entry.push((edge.to, edge_idx));
            }
        }
        edge_lookup
    }
    pub fn get_input_size(&self) -> usize {
        self.input_size
    }
    pub fn get_output_size(&self) -> usize {
        self.output_size
    }
    /**returns new instance along with new innovation number*/
    pub fn new(input_size: usize, output_size: usize, mut innovation_no: usize) -> (Self, usize) {
        let mut nodes = vec![Node { activation: None }; input_size];
        nodes.append(&mut vec![Node { activation: Some(activations::IDENTITY) }; output_size]);
        let mut edges = Vec::with_capacity(output_size.max(input_size));
        if input_size > output_size {
            for (dst_node, src_node) in (0..input_size).enumerate() {
                let dst_node = input_size + dst_node % output_size;
                innovation_no += 1;
                edges.push(Edge { from: src_node, weight: X::random(), to: dst_node, enabled: true, innovation_no })
            }
        } else {
            for (src_node, dst_node) in (input_size..(input_size + output_size)).enumerate() {
                let src_node = src_node % input_size;
                innovation_no += 1;
                edges.push(Edge { from: src_node, weight: X::random(), to: dst_node, enabled: true, innovation_no })
            }
        }
        let s = Self { nodes, edges, input_size, output_size };
        s.assert_invariants();
        assert!(s.is_acyclic());
        (s, innovation_no)
    }
    /**Checks if graphs is acyclic. Usually it should be, unless you're
     trying to evolve recurrent neural networks.*/
    pub fn is_acyclic(&self) -> bool {
        let mut visited = vec![false; self.nodes.len()];
        let mut on_stack = vec![false; self.nodes.len()];
        let mut min_unvisited = 0;
        let lookup = self.build_edge_lookup_table();
        while let Some(unvisited_idx) = visited[min_unvisited..].iter().position(|&x| !x) {
            let unvisited_idx = min_unvisited + unvisited_idx;
            assert!(on_stack.iter().all(|&x| !x));
            assert!(!visited[unvisited_idx]);
            min_unvisited = unvisited_idx + 1;
            fn has_cycle(current_idx: usize, lookup: &Vec<Vec<(usize, usize)>>, visited: &mut Vec<bool>, on_stack: &mut Vec<bool>) -> bool {
                visited[current_idx] = true;
                on_stack[current_idx] = true;
                for &(outgoing, _) in &lookup[current_idx] {
                    if !visited[outgoing] {
                        if has_cycle(outgoing, lookup, visited, on_stack) {
                            return true;
                        }
                    } else if on_stack[outgoing] {
                        return true;
                    }
                }
                on_stack[current_idx] = false;
                false
            }
            if has_cycle(unvisited_idx, &lookup, &mut visited, &mut on_stack) {
                return false;
            }
        }
        true
    }
    /**Check if introduction of such connection would break the assumption of acyclicity.
    The network must be acyclic in order to be convertible into feed-forward neural net.
    This function only works under the precondition that the graph is initially acyclic.*/
    pub fn can_connect(&self, from: usize, to: usize) -> bool {
        if from == to { return false; }
        if to < self.input_size { return false; }
        assert!(self.is_acyclic(), "{}", self);
        let mut stack = Vec::<usize>::new();
        let lookup = self.build_edge_lookup_table();
        stack.push(to);
        while let Some(src_node) = stack.pop() {
            for &(dst_node, _) in lookup.get(src_node).unwrap() {
                if dst_node == from {
                    return false;
                } else {
                    stack.push(dst_node);
                }
            }
        }
        true
    }

    /**Will introduce connection only if it doesn't break acyclicity. If successful, increases innovation number and returns its new value.
    Testing whether connection was successfully added, can be easily achieved by checking if old value of innovation number is different from
    the returned one.*/
    pub fn add_connection_if_possible(&mut self, from: usize, to: usize, weight: X, innovation_no: usize) -> usize {
        if self.can_connect(from, to) {
            let inno = self.add_connection_forcefully(from, to, weight, innovation_no);
            assert!(self.is_acyclic(), "{}", self);
            inno
        } else {
            innovation_no
        }
    }
    /**Will introduce connection. The user is in charge of making sure that no cycle would be
    introduced (unless you're building a recurrent network - then feel free to go ahead
    without any checks). This function returns an incremented innovation number.
    */
    pub fn add_connection_forcefully(&mut self, from: usize, to: usize, weight: X, mut innovation_no: usize) -> usize {
        innovation_no += 1;
        self.edges.push(Edge {
            innovation_no,
            enabled: true,
            from,
            weight,
            to,
        });
        self.assert_invariants();
        innovation_no
    }
    pub fn search_connection_by_endpoints(&mut self, from: usize, to: usize) -> Option<usize> {
        self.edges.iter().position(|e| e.from == from && e.to == to)
    }
    pub fn assert_invariants(&self) {
        assert!(self.edges.windows(2).all(|e| e[0].innovation_no < e[1].innovation_no), "Edges are not sorted by innovation number:\n{}", self);
        let nodes = self.node_count();
        assert!(self.edges.iter().all(|e| e.to < nodes), "Destination of edge points to non-existent node:\n{}", self);
        assert!(self.edges.iter().all(|e| e.from < nodes), "Source of edge points to non-existent node:\n{}", self);
    }
    pub fn get_random_node(&self) -> usize {
        self.node_count().random()
    }
    pub fn get_random_non_input_node(&self) -> usize {
        self.input_size + (self.node_count() - self.input_size).random()
    }
    pub fn get_random_edge(&self) -> usize {
        self.edge_count().random()
    }
    /**Splits an edge in half and introduces a new node in the middle. As a side effect, two
    new connections are added (representing the two halves of old edge) and
    the original edge becomes disabled. Two new innovation numbers are added.
    Returns new innovation number.*/
    pub fn add_node(&mut self, edge_index: usize, activation: &'static ActFn, mut innovation_no: usize) -> usize {
        let was_acyclic = self.is_acyclic();
        let from = self.edges[edge_index].from;
        let to = self.edges[edge_index].to;
        let weight = self.edges[edge_index].weight;
        let new_node_idx = self.nodes.len();
        innovation_no += 1;
        let incoming_edge = Edge {
            innovation_no,
            enabled: true,
            from,
            weight,
            to: new_node_idx,
        };
        innovation_no += 1;
        let outgoing_edge = Edge {
            innovation_no,
            enabled: true,
            from: new_node_idx,
            weight,
            to,
        };
        self.nodes.push(Node {
            activation: Some(activation),
        });
        self.edges.push(incoming_edge);
        self.edges.push(outgoing_edge);
        self.edges[edge_index].enabled = false;
        self.assert_invariants();
        assert_eq!(was_acyclic, self.is_acyclic());
        innovation_no
    }
    pub fn get_activation(&mut self, node_idx: usize) -> Option<&'static ActFn> {
        self.nodes[node_idx].activation
    }
    /**Sets new activation function for a node. If the node is an input node, then it has no
    activation and hence it cannot be changed. Returns true if change was successful,
    false if it was input node that couldn't be mutated.*/
    pub fn set_activation(&mut self, node_idx: usize, f: &'static ActFn) -> bool {
        match &mut self.nodes[node_idx].activation {
            None => { false }
            Some(old) => {
                *old = f;
                true
            }
        }
    }
    pub fn set_weight(&mut self, edge_idx: usize, weight: X) {
        self.edges[edge_idx].weight = weight
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
    pub fn edge_src(&self, edge_idx: usize) -> usize {
        self.edges[edge_idx].from
    }
    pub fn edge_dest(&self, edge_idx: usize) -> usize {
        self.edges[edge_idx].to
    }
    pub fn edge_innovation_no(&self, edge_idx: usize) -> usize {
        self.edges[edge_idx].innovation_no
    }
    pub fn get_weight(&self, edge_idx: usize) -> X {
        self.edges[edge_idx].weight
    }
    pub fn is_enabled(&self, edge_idx: usize) -> bool {
        self.edges[edge_idx].enabled
    }
    pub fn set_enabled(&mut self, edge_idx: usize, enabled: bool) {
        self.edges[edge_idx].enabled = enabled
    }
    /**It is assumed that this network is more fit than the other and hence it will retain all
    it's original connections.
    */
    pub fn crossover_in_place(&mut self, other: &Self) {
        let was_acyclic = self.is_acyclic();
        let mut j = 0; // index of edge in the other

        'outer: for edge in self.edges.iter_mut() {
            if f32::random() < 0.5f32 { // sometimes edge weight will be crossed-over and sometimes not
                while other.edges[j].innovation_no < edge.innovation_no {
                    j += 1;
                    if j >= other.edges.len() {
                        break 'outer;
                    }
                }
                if other.edges[j].innovation_no == edge.innovation_no {
                    edge.weight = other.edges[j].weight;
                }
            }
        }
        for (my_node, other_node) in self.nodes.iter_mut().zip(other.nodes.iter()) {
            // Note that because all the edges are crossed over according to their innovation number,
            // the source and destination nodes remain unaffected by such operation
            // (equal innovation number means equal source and destination). Therefore
            // we can cross-over the nodes as well, without much risk (it is still possible to
            // cross-over two unrelated nodes, but it's less likely)
            if f32::random() < 0.5f32 {
                my_node.activation = other_node.activation.clone();
            }
        }
        self.assert_invariants();
        assert_eq!(was_acyclic, self.is_acyclic(), "{}", self)
    }

    pub fn crossover(&self, other: &Self) -> Self {
        let mut c = self.clone();
        c.crossover_in_place(other);
        c
    }
    /**Returns node indices sorted in topological order and it also returns a lookup
    table of outgoing edges as a by-product*/
    fn topological_sort(&self) -> (Vec<usize>, Vec<Vec<(usize, usize)>>) {
        assert!(self.is_acyclic(), "{}", self);
        let mut visited = vec![false; self.nodes.len()];
        let mut min_unvisited_idx = 0;
        let mut topological_order = Vec::new();
        let lookup = self.build_enabled_edge_only_lookup_table();
        while let Some(unvisited_idx) = visited[min_unvisited_idx..].iter().position(|&x| !x) {
            let unvisited_idx = min_unvisited_idx + unvisited_idx;
            min_unvisited_idx = unvisited_idx + 1;
            fn rec(lookup: &Vec<Vec<(usize, usize)>>, visited: &mut Vec<bool>,
                   topological_order: &mut Vec<usize>, node: usize) {
                for &(dst_node, _) in &lookup[node] {
                    if !visited[dst_node] {
                        rec(lookup, visited, topological_order, dst_node);
                    }
                }
                assert!(!topological_order.contains(&node));
                assert!(!visited[node]);
                topological_order.push(node);
                visited[node] = true;
            }
            rec(&lookup, &mut visited, &mut topological_order, unvisited_idx);
        }
        assert!(self.edges.iter().all(|edge|
            topological_order.iter().position(|&n| n == edge.to) <
                topological_order.iter().position(|&n| n == edge.from)));
        (topological_order, lookup)
    }
    /**The the current genotype (the CPPN) and compile it into a phenotype (feed-forward network)*/
    pub fn build_feed_forward_net(&self) -> FeedForwardNet<X> {
        let (topological_order, lookup) = self.topological_sort();
        let mut instructions = Vec::with_capacity(self.edges.len() + self.nodes.len());
        for &node_idx in topological_order.iter().rev() {
            let node = &self.nodes[node_idx];
            if let Some(f) = node.activation {
                if !f.is_identity() { // a small optimisation
                    instructions.push(EdgeOrNode::Node(node_idx, f));
                }
            }
            for &(_, outgoing_edge_idx) in &lookup[node_idx] {
                let edge = &self.edges[outgoing_edge_idx];
                if edge.enabled {
                    instructions.push(EdgeOrNode::Edge(edge.from, edge.weight, edge.to))
                }
            }
        }
        FeedForwardNet {
            net: instructions,
            len: self.node_count(),
            input_size: self.input_size,
            output_size: self.output_size,
        }
    }
}

impl<X: Num> Display for CPPN<X> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (idx, node) in self.nodes.iter().enumerate() {
            if let Some(a_f) = node.activation {
                if idx < self.input_size + self.output_size {
                    writeln!(f, "Output({}) node {} is {}", idx - self.input_size, idx, a_f.name());
                } else {
                    writeln!(f, "Node {} is {}", idx, a_f.name());
                }
            } else {
                assert!(idx < self.input_size);
                writeln!(f, "Input node {}", idx);
            }
        }
        for (idx, edge) in self.edges.iter().enumerate() {
            writeln!(f, "{} {} from {} to {} with weight {} and innovation number {}", if edge.enabled { "Edge" } else { "Disabled edge" }, idx, edge.from, edge.to, edge.weight, edge.innovation_no);
        }
        Ok(())
    }
}

