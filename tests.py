import random
import rusty_neat

assert rusty_neat.activation_functions() == ["sigmoid", "relu", "sin", "cos", "tan", "tanh", "abs", "identity"]

input_neurons = 4
output_neurons = 3

# First initialise NEAT.
# Neat64 operates on double-precision floating points,
# while Neat32 uses single-precision floats

neat = rusty_neat.Neat64(input_neurons, output_neurons)

# Optionally you may specify a list of activation functions

neat1 = rusty_neat.Neat64(input_neurons, output_neurons,
                          ["sigmoid", "relu", "sin", "cos", "tan", "tanh", "abs", "identity"])

# You can lookup the global innovation number (initially 0)
assert neat.current_innovation_number == 0

# generate new blank CPPN
cppn = neat.new_cppn()

# Every creation of CPPN will increase innovation number
inno_num_after_new_cppn = neat.current_innovation_number
assert inno_num_after_new_cppn > 0

# compile CPPN to feed-forward network
net = cppn.build_feed_forward_net()

# Before running the network you need to prepare buffer that will hold
# activation values of all neurons (here the feed-forward network is not divided into layers but
# instead it is an arbitrary directed acyclic graph and its neurons are evaluated in topological order)
out = net.make_output_buffer()

# write some input
out.set_input([1, 2, 3, 4])

# You can later query the input as well
assert out.get_input() == [1, 2, 3, 4]

# By default the output is set to zero
random_output_before = out.get_output()
assert random_output_before == [0, 0, 0]

# but if you run the network
net(out)

# get the output will change
actual_output = out.get_output()
assert actual_output[0] != 0 and actual_output[1] != 0 and actual_output[2] != 0

# You can see the total number of neurons in a network
assert cppn.node_count() == input_neurons+output_neurons

# There are also several edges
assert cppn.edge_count() > 0

prev_inno = -1
# You can iterate those edges and query their information
for edge_index in range(cppn.edge_count()):
    # The source of edge
    assert cppn.edge_src_node(edge_index) < cppn.node_count()
    # The destination of edge
    assert cppn.edge_dest_node(edge_index) < cppn.node_count()
    # The innovation number of an edge
    inno = cppn.edge_innovation_number(edge_index)
    # The edges are sorted by innovation number in increasing order
    # (This makes crossover operation easier to carry out)
    assert inno > prev_inno
    prev_inno = inno
    # You cannot lookup activation function of any edge, because it's stored
    # in a non-readable format, but you can change it
    cppn.set_activation_function(edge_index, random.choice(rusty_neat.activation_functions()))
    # There is even a shorthand
    cppn.set_activation_function(edge_index, rusty_neat.random_activation_fn())
    # but it's recommended to use
    neat.set_random_activation_function(cppn, edge_index)
    # (This way we restrict the possible choices only to the functions provided to neat instance,
    # instead of the global list of all implemented activations)
    #
    # You can set weight of connection is a similar manner
    cppn.set_weight(edge_index, 2.3)
    # Or alternatively
    cppn.set_weight(edge_index, neat.random_weight())

# It's possible choose an edge randomly
edge_index = cppn.get_random_edge()
# and then split it in half by adding a new node in the middle (using add_node function).
# The index of newly added node will be equal to the current number of nodes
node_count = cppn.node_count()
new_node_index = node_count
# The global innovation number will also be affected
prev_inno = neat.current_innovation_number
neat.add_node(cppn, edge_index)  # splits edge in two
assert prev_inno < neat.current_innovation_number  # add_node increases innovation number
# Now the number of edges has increased by one
node_count = node_count+1
assert node_count == cppn.node_count()

# we can add a new connection to the newly added node
source_node = 0  # first input node
destination_node = new_node_index
prev_inno = neat.current_innovation_number
was_successful = neat.add_connection(cppn, source_node, destination_node)
# This operation could fail, if introduction of such edge leads to recurrent cycle
# in the network (remember that it must be convertible to feed-forward network!)
assert was_successful  # In this simple case we can be sure that it was ok
if was_successful:
    # add_connection increases innovation number but only if it was successfully carried out
    assert prev_inno < neat.current_innovation_number
# Rather than initialising CPPN individually
# (each one with different innovation number),
# you can also initialise them in batches
# (and all of them will share the same structure and the same innovation
# numbers but with different random weights on edges)
population_size = 16
assert neat1.current_innovation_number == 0
population = neat1.new_cppns(population_size)
# All CPPNs share the same innovation numbers so the global number is the same as if
# we initialised only one CPPN.
assert inno_num_after_new_cppn == neat1.current_innovation_number

# Now that we have multiple CPPNs we can randomly mutate them in a batch
NODE_ADDITION_PROB = 0.1
EDGE_ADDITION_PROB = 0.1
ACTIVATION_MUTATION_PROB = 0.1
WEIGHT_MUTATION_PROB = 0.1
for cppn in population:
    if random.random() < NODE_ADDITION_PROB:
        neat1.add_random_node(cppn)
        # This is equivalent to
        # neat1.add_node(cppn, cppn.get_random_edge())
    if random.random() < EDGE_ADDITION_PROB:
        was_successful = neat1.add_random_connection(cppn)  # Notice that it may fail
        # This is equivalent to
        # was_successful = neat1.add_connection(cppn, cppn.get_random_node(), cppn.get_random_node())
    for edge_index in range(cppn.edge_count()):
        if random.random() < ACTIVATION_MUTATION_PROB:
            neat1.set_random_activation_function(cppn, edge_index)
        if random.random() < WEIGHT_MUTATION_PROB:
            neat1.set_random_activation_function(cppn, edge_index)
            cppn.set_weight(edge_index, neat1.random_weight())

