#!/bin/bash
set -e
export RUST_BACKTRACE=1
cargo build
mv target/debug/librusty_neat.dylib rusty_neat.so
python3 tests.py
