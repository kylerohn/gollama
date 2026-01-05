# gollama

An LLM inference engine/framework in Go built off llama.cpp

## Overview

## Dependencies

### Go
- Go 1.24 +

### Build toolchain
- CMake
- C/C++ compiler (gcc/clang)
- Make or Ninja
  
### OS
- Only tested on Ubuntu 22.04 and 24.04
- On Windows use WSL

## Install/Build

This is not an official Go library (yet), and must be cloned manually.

The code in this repository relies on the following submodules:
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

### Clone Repo

```
git clone --recuse-submodules https://github.com/kylerohn/gollama.git
```

#### OR

```
git clone https://github.com/kylerohn/gollama.git
git submodule update --init --recursive
```

### Build Backend

gollama requires llama.cpp to be built to run, which uses go generate:

```
go generate ./...
```

by default, gollama is built for running on CPU, with CMake using Makefiles, but the following environment variables can be set to change the behavior:

generators:
```
LLAMA_GENERATOR=Ninja
```

backends:
```
LLAMA_BACKEND=cuda
```