
# Biological and Machine Intelligence Library

A Python library written in Rust. Provides various tools for biologically-inspired machine learning.



### NEAT genetic algorithms 

#### Implemented features:

- Compositional Pattern Producing Networks (CPPN) with crossover
- Picbreeder 
- HyperNEAT 
- numpy integration
- GPU acceleration (with OpenCL)

#### Planned:

- ES-HyperNEAT
- Plastic Hyper-HEAT
- Continous-time recurrent neural networks (CTRNN)
- Novelty Search
- L-systems producing deep fractal neural networks
- L-systems producing plastic neural networks

### HTM biologically-constrained neural networks

#### Implemented features:

- Sparse distributed representation (SDR)
- SDR encoders 
- Spacial Poolers
- Pattern separation with negative spacial poolers
- GPU acceleration with OpenCL 
- Higher order memory with the temporal memory algorithm

#### Planned:

## Building

```
cargo build --release
```

Then you can find produced artifacts in `target/release`.

While developing, you can symlink (or copy) and rename the shared 
library from the target folder: On MacOS, rename 
`librusty_neat.dylib` to `rusty_neat.so`, on Windows `librusty_neat.dll` 
to `rusty_neat.pyd`, and on Linux `librusty_neat.so` to `rusty_neat.so`. 
Then open a Python shell in the same folder and you'll be able to `import rusty_neat`.




## Usage:

Check out this [tutorial](rusty_neat_quick_guide.py) for details.


