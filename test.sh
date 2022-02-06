#!/bin/bash
set -e
export RUST_BACKTRACE=1
cargo build --release
mv target/release/librusty_neat.dylib rusty_neat.so
cp rusty_neat.so experiments/
python3 ./rusty_neat_quick_guide.py
python3 ./ndalgebra_tests.py

