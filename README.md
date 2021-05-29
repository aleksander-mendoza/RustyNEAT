
# RustyNEAT

A Python library written in Rust

### Implemented features:

- Compositional Pattern Producing Networks (CPPN) with crossover

### Planned:

- HyperNEAT
- ES-HyperNEAT
- Plastic Hyper-HEAT
- Continous-time recurrent neural networks (CTRNN)
- Novelty Search
- L-systems producing deep fractal neural networks
- L-systems producing plastic neural networks

### Building

```
cargo build --release
```

Then you can find produced artifacts in `target/release`.

While developing, you can symlink (or copy) and rename the shared 
library from the target folder: On MacOS, rename 
`librusty_neat.dylib` to `rusty_neat.so`, on Windows `librusty_neat.dll` 
to `rusty_neat.pyd`, and on Linux `librusty_neat.so` to `rusty_neat.so`. 
Then open a Python shell in the same folder and you'll be able to `import rusty_neat`.




### Usage:

```
import rusty_neat

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

# run the network
net.run(out)

# get the output
some_list = out.get_output()
 
```

Note that CPPN are not the right tool if you wish to evolve large neural networks.
Instead you may compile a CPPN to a dense layer of larger neural network using HyperHEAT.
Alternatively you may use L-systems to evolve fractal neural networks.
