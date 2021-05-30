#!/bin/bash
cargo build --release
mv target/release/librusty_neat.dylib rusty_neat.so
python3 tests.py
