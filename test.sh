#!/bin/bash
set -e
cargo build
mv target/debug/librusty_neat.dylib rusty_neat.so
python3 tests.py
