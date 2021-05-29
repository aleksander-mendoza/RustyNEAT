use crate::num::Num;
use crate::util::Initializer;
use rand::Rng;
use crate::activations;


enum EdgeOrNode<X> {
    Node(usize, fn(X) -> X),
    Edge(usize, X, usize),
}

pub struct FeedForwardNet<X: Num> {
    net: Vec<EdgeOrNode<X>>,
    len: usize,
}

impl<X: Num> FeedForwardNet<X> {
    pub fn new_input_buffer(&self) -> Vec<X> {
        vec![X::zero(); self.len]
    }
    pub fn run(&self, input_buffer: &mut [X]) {
        assert!(input_buffer.len() >= self.len);
        for instruction in &self.net {
            match *instruction {
                EdgeOrNode::Edge(from, w, to) => input_buffer[to] = input_buffer[to] + w * input_buffer[from],
                EdgeOrNode::Node(from, activation) => input_buffer[from] = activation(input_buffer[from]),
            }
        }
    }
}




#[derive(Clone)]
struct Node<X: Num> {
    /**Initial nodes do not have any activation*/
    activation: Option<fn(X) -> X>,
}

#[derive(Clone)]
pub struct CPPN<X: Num> {
    nodes: Vec<Node<X>>,
    edges: Vec<Edge<X>>,
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
    fn build_edge_lookup_table(&self) -> Vec<Vec<(usize,usize)>> {
        let mut edge_lookup = Vec::initialize(self.nodes.len(), |_| Vec::new());
        for (edge_idx,edge) in self.edges.iter().enumerate() {
            let mut entry = &mut edge_lookup[edge.from];
            // assert!(!entry.contains(&edge.to));
            if edge.enabled {
                entry.push((edge.to,edge_idx));
            }
        }
        edge_lookup
    }

    /**returns new instance along with new innovation number*/
    pub fn new<F: FnMut() -> X>(input_size: usize, output_size: usize, mut innovation_no: usize, mut weight_generator: F) -> (Self, usize) {
        let mut nodes = vec![Node { activation: None }; input_size];
        nodes.append(&mut vec![Node { activation: Some(activations::identity) }; output_size]);
        let mut edges = Vec::with_capacity(output_size.max(input_size));
        if input_size > output_size {
            for (dst_node, src_node) in (0..input_size).enumerate() {
                let dst_node = dst_node % output_size;
                innovation_no += 1;
                edges.push(Edge { from: src_node, weight: weight_generator(), to: dst_node, enabled: true, innovation_no })
            }
        } else {
            for (src_node, dst_node) in (input_size..(input_size + output_size)).enumerate() {
                let src_node = src_node % input_size;
                innovation_no += 1;
                edges.push(Edge { from: src_node, weight: weight_generator(), to: dst_node, enabled: true, innovation_no })
            }
        }
        (Self { nodes, edges }, innovation_no)
    }
    /**Checks if graphs is acyclic. Usually it should be, unless you're
     trying to evolve recurrent neural networks.*/
    pub fn is_acyclic(&self) -> bool {
        let mut visited = vec![false; self.nodes.len()];
        let mut min_unvisited = 0;
        let mut stack = Vec::<usize>::new();
        let lookup = self.build_edge_lookup_table();
        while let Some(unvisited_idx) = visited[min_unvisited..].iter().position(|&x| !x) {
            min_unvisited = unvisited_idx + 1;
            stack.push(unvisited_idx);
            while let Some(src_node) = stack.pop() {
                visited[src_node] = true;
                for &(dst_node,_) in lookup.get(src_node).unwrap() {
                    if visited[dst_node] {
                        return false;
                    } else {
                        stack.push(dst_node);
                    }
                }
            }
        }
        true
    }
    /**Check if introduction of such connection would break the assumption of acyclicity.
    The network must be acyclic in order to be convertible into feed-forward neural net.
    This function only works under the precondition that the graph is initially acyclic.*/
    pub fn can_connect(&self, from: usize, to: usize) -> bool {
        assert!(self.is_acyclic());
        let mut stack = Vec::<usize>::new();
        let lookup = self.build_edge_lookup_table();
        stack.push(to);
        while let Some(src_node) = stack.pop() {
            for &(dst_node,_) in lookup.get(src_node).unwrap() {
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
    pub fn add_connection_if_possible(&mut self, from: usize, to: usize, weight: X, mut innovation_no: usize) -> usize {
        if self.can_connect(from, to) {
            self.add_connection_forcefully(from, to, weight, innovation_no)
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
    fn assert_invariants(&self) {
        assert!(self.edges.windows(2).all(|e| e[0].innovation_no < e[1].innovation_no));
    }
    /**Splits an edge in half and introduces a new node in the middle. As a side effect, two
    new connections are added (representing the two halves of old edge) and
    the original edge becomes disabled. Two new innovation numbers are added.
    Returns new innovation number.*/
    pub fn add_node(&mut self, edge_index: usize, activation: fn(X) -> X, mut innovation_no: usize) -> usize {
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
        innovation_no
    }
    pub fn has_activation(&mut self, node_idx: usize) -> bool {
        self.nodes[node_idx].activation == None
    }
    pub fn set_activation(&mut self, node_idx: usize, f: fn(X) -> X) {
        self.nodes[node_idx].activation = Some(f)
    }
    pub fn replace_activation_if_present(&mut self, node_idx: usize, f: fn(X) -> X) -> bool {
        if self.has_activation(node_idx) {
            self.set_activation(node_idx, f);
            true
        } else {
            false
        }
    }
    pub fn set_weight(&mut self, edge_idx: usize, weight: X) {
        self.edges[edge_idx].weight = weight
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
    /**It is assumed that this network is more fit than the other and hence it will retain.
    */
    pub fn crossover_in_place<R: Rng>(&mut self, other: &Self, rng: &mut R) {
        let mut j = 0; // index of edge in other

        for edge in self.edges.iter_mut() {
            if rng.gen_bool(0.5) {
                while other.edges[j].innovation_no < edge.innovation_no {
                    j += 1;
                }
                if other.edges[j].innovation_no == edge.innovation_no {
                    edge.weight = other.edges[j].weight;
                }
            }
        }
    }
    pub fn crossover<R: FnMut()->X>(&self, other: &Self, rng: &mut R) -> Self {
        let c = self.clone();
        c.crossover(other, rng);
        c
    }
    /**Returns node indices sorted in topological order and it also returns a lookup
    table of outgoing edges as a by-product*/
    fn topological_sort(&self) -> (Vec<usize>, Vec<Vec<(usize,usize)>>) {
        let mut visited = vec![false; self.nodes.len()];
        let mut min_unvisited_idx = 0;
        let mut topological_order = Vec::new();
        let lookup = self.build_edge_lookup_table();
        while let Some(unvisited_idx) = visited[min_unvisited_idx..].iter().position(|&x| !x) {
            let unvisited_idx = min_unvisited_idx + unvisited_idx;
            min_unvisited_idx = unvisited_idx + 1;
            fn rec(lookup: &Vec<Vec<(usize,usize)>>, visited: &mut Vec<bool>,
                   topological_order: &mut Vec<usize>, node: usize) {
                for &(dst_node,_) in &lookup[node] {
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
                instructions.push(EdgeOrNode::Node(node_idx, f));
            }
            for &(_,outgoing_edge_idx) in &lookup[node_idx] {
                let edge = &self.edges[outgoing_edge_idx];
                if edge.enabled {
                    instructions.push(EdgeOrNode::Edge(edge.from, edge.weight, edge.to))
                }
            }
        }
        FeedForwardNet { net: instructions, len: self.nodes.len() }
    }
}

