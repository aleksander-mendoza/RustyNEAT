import random
import rusty_neat

assert rusty_neat.activation_functions() == ["identity", "sigmoid", "relu", "sin", "cos", "tan", "tanh", "abs",
                                             "square", "inv", "step"]

input_neurons = 4
output_neurons = 3

# First initialise NEAT.
# Neat64 operates on double-precision floating points,
# while Neat32 uses single-precision floats

neat = rusty_neat.Neat32(input_neurons, output_neurons)

# Optionally you may specify a list of activation functions

neat1 = rusty_neat.Neat32(input_neurons, output_neurons,
                          ["sigmoid", "relu", "sin", "cos", "tan", "tanh", "abs", "identity"])

# You can later lookup functions used by NEAT
assert neat1.get_activation_functions() == ["sigmoid", "relu", "sin", "cos", "tan", "tanh", "abs", "identity"];

# By default all activation functions are allowed
assert neat.get_activation_functions() == rusty_neat.activation_functions()

# You can lookup the global innovation number (initially 0)
assert neat.current_innovation_number == 0
# You are not allowed to manually change it though

# Neat can be used to generate new random CPPN
cppn = neat.new_cppn()  # This CPPN is has the minimal necessary number of nodes and edges

# Every creation of new CPPN will increase innovation number
inno_num_after_new_cppn = neat.current_innovation_number
assert inno_num_after_new_cppn > 0

# compile CPPN to feed-forward network
net = cppn.build_feed_forward_net()

# you can run the network
out = net([1, 2, 3, 4])

# get the output will change
assert out[0] != 0 and out[1] != 0 and out[2] != 0

# You can see the total number of neurons in a network
assert cppn.node_count() == input_neurons + output_neurons

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
    assert inno > prev_inno  # The edges are sorted by innovation number in increasing order
    # (This makes crossover operation easier to carry out)
    prev_inno = inno
    # You can set weight of connection is a similar manner
    cppn.set_weight(edge_index, 2.3)
    # Or alternatively
    cppn.set_weight(edge_index, neat.random_weight())
    # Every edge may be enabled or disabled. Edges are never fully removed from
    # the network, because it would lead to losing information about innovation numbers
    # (especially if later mutation re-added the same weight). This would make crossover
    # more difficult. So the only way to get rid of an edge is by disabling it
    cppn.set_enabled(edge_index, False)
    # You can check if edge is enabled or not by using
    assert not cppn.is_enabled(edge_index)
    # You can later re-enable an edge with
    cppn.set_enabled(edge_index, True)
    assert cppn.is_enabled(edge_index)

# You can iterate nodes in a similar way
for node_index in range(cppn.node_count()):
    # You can lookup activation function of any node
    act_fn = cppn.get_activation_function(node_index)
    if node_index < cppn.input_size:  # input nodes do not have any activations
        assert act_fn is None
    else:
        assert act_fn is not None
    new_act_fn = random.choice(rusty_neat.activation_functions())
    was_successful = cppn.set_activation_function(node_index, new_act_fn)
    act_fn = cppn.get_activation_function(node_index)
    if node_index < cppn.input_size:
        assert not was_successful
        assert act_fn is None  # You cannot assign activation function to input node
    else:
        assert was_successful
        assert act_fn == new_act_fn
    # There is even a shorthand
    cppn.set_activation_function(node_index, rusty_neat.random_activation_fn())
    # but it's recommended to use
    neat.set_random_activation_function(cppn, node_index)
    # (This way we restrict the possible choices only to the functions provided to neat instance,
    # instead of the global list of all implemented activations)

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
node_count = node_count + 1
assert node_count == cppn.node_count()
# You can verify that the original edge is now disabled
assert not cppn.is_enabled(edge_index)
# We can query an edge connecting two nodes (keep in mind that it's a linear search operation)
incoming_to_new_node = cppn.search_connection_by_endpoints(cppn.edge_src_node(edge_index), new_node_index)
outgoing_from_new_node = cppn.search_connection_by_endpoints(new_node_index, cppn.edge_dest_node(edge_index))
assert cppn.is_enabled(incoming_to_new_node)
assert cppn.is_enabled(outgoing_from_new_node)

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
NODE_ACTIVATION_FN_MUTATION_PROB = 0.1
WEIGHT_MUTATION_PROB = 0.1
EDGE_ACTIVATION_FLIP_PROB = 0.1
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
        if random.random() < WEIGHT_MUTATION_PROB:
            cppn.set_weight(edge_index, neat1.random_weight())
    for node_idx in range(cppn.node_count()):
        if random.random() < NODE_ACTIVATION_FN_MUTATION_PROB:
            neat1.set_random_activation_function(cppn, node_idx)

# This entire loop above could be shortened to just a single (slightly faster) equivalent call
neat.mutate_population(population,
                       NODE_ADDITION_PROB,
                       EDGE_ADDITION_PROB,
                       NODE_ACTIVATION_FN_MUTATION_PROB,
                       WEIGHT_MUTATION_PROB)

# Aside from mutations, there is also the possibility of performing cross-over.
# First choose one individual that is deemed to be fitter.
fitter_cppn = population[0]  # index zero was chosen just for this example,
# but normally you would have some fitness function.
# Next we need another, less fit individual
less_fit_cppn = population[1]
child = fitter_cppn.crossover(less_fit_cppn)

# These are all the essential functions for manipulating CPPNs.
# We can try to apply them to a simple toy problem. Let's learn
# a neural network that solves the XOR problem.

neat = rusty_neat.Neat32(2, 1,  # two input bits and output XORed bit
                         ["sigmoid", "identity", "relu", "sin", "cos", "tan", "tanh", "abs"])
XOR_TABLE = [
    [0, 0, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0],
]


def fitness(cppn) -> float:
    net = cppn.build_feed_forward_net()
    loss = 0
    for in1, in2, expected in XOR_TABLE:
        loss += (expected - net([in1, in2])[0]) ** 2
    fitness_val = -loss  # we want that higher values mean better fitness
    return fitness_val


population = neat.new_cppns(16)
TO_ELIMINATE = 8  # Number of weakest CPPNs to eliminate at each generation and replace with crossover of other CPPNs
for generation in range(100):
    evaluated = [(fitness(cppn), cppn) for cppn in population]
    evaluated.sort(key=lambda x: x[0])  # sort by fitness in ascending order
    total_fitness = sum(map(lambda x: x[0], evaluated))
    max_fitness = max(map(lambda x: x[0], evaluated))
    average_fitness = total_fitness / len(population)
    print("Generation=" + str(generation) +
          ", avg fitness=" + str(average_fitness) +
          ", max fitness=" + str(max_fitness))
    for i in range(TO_ELIMINATE):  # replace first few CPPNs with new ones
        a = random.randrange(0, len(population))
        b = random.randrange(0, len(population))
        less_fit_cppn, fitter_cppn = population[min(a, b)], population[max(a, b)]
        population[i] = fitter_cppn.crossover(less_fit_cppn)
    neat.mutate_population(population, 0.1, 0.1, 0.1, 0.1)

assert max_fitness > -2  # If everything is implemented correctly, this result should be achieved with ease
# This concludes the short end-to-end example of evolution algorithm with RustyNEAT. Next we will move on to
# parallel processing.

# RustyNEAT comes with GPU support. What's more is that instead of CUDA, it uses
# OpenCL as it's backend and can work with any hardware. You can query installed OpenCL
# platforms using
platform_list = rusty_neat.platforms()
assert len(platform_list) > 0, "No OpenCL available!"

# You can also get a list of all available computing devices (both GPUs and CPUs)
device_list = rusty_neat.devices()
assert len(device_list) > 0

# If you have multiple versions of OpenCL installed, then you may query devices
# accessible to a specific version
assert list(map(str, rusty_neat.devices(platform_list[0]))) == list(map(str, rusty_neat.devices()))

