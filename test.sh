#!/bin/bash
set -e
export RUST_BACKTRACE=1
cargo build
mv target/debug/librusty_neat.dylib rusty_neat.so
python3 ./ndalgebra_tests.py
python3 ./rusty_neat_quick_guide.py
