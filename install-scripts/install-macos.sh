#!/usr/bin/env bash
set -ex

# Install Rust and Cargo
curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"

# Make sure Rust has been installed correctly
rustc --version

# Install pyenv
brew install pyenv gmp

# Make sure pyenv has been installed correctly
pyenv -v

make deps-macos

pyenv local 3.9.15

pip install -r requirements.txt

echo "-- You need to follow these instructions to finish installing pyenv: --"
pyenv init || true
