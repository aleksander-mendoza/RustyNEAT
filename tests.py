import rusty_neat

assert rusty_neat.activation_functions() == ["sigmoid", "relu", "sin", "cos", "tan", "tanh", "abs", "identity"]

input_neurons = 4
output_neurons = 3

# First initialise NEAT.
# Neat64 operates on double-precision floating points,
# while Neat32 uses single-precision floats

neat = rusty_neat.Neat64(input_neurons, output_neurons)

# Optionally you may specify a list of activation functions

neat = rusty_neat.Neat64(input_neurons, output_neurons, ["sigmoid", "relu", "sin", "cos", "tan", "tanh", "abs", "identity"])

# generate new blank CPPN
cppn = neat.new_cppn()

# compile CPPN to feed-forward network
net = cppn.build_feed_forward_net()

# prepare output buffer
out = net.make_output_buffer()

# write some input
out.set_input([1,2,3,4])
assert out.get_input() == [1,2,3,4]
random_output_before = out.get_output()
assert len(random_output_before) == output_neurons
# run the network
net(out)

# get the output
actual_output = out.get_output()

assert random_output_before != actual_output

# Rather than initialising CPPN individually
# (each one with different innovation number),
# you can also initialise them in batches
# (and all of them will share the same structure and the same innovation numbers but with different random weights on edges)
population_size = 16
population = neat.new_cppns(population_size)

