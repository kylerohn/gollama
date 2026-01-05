# gollama

An LLM inference engine/framework in Go built on top of llama.cpp.

## Overview

gollama is still in development and the end goal isn't entirely clear at this point

## Dependencies

### Go
- Go 1.24+

### Build toolchain
- CMake
- C/C++ compiler (gcc or clang)
- Make or Ninja

### OS
- Tested on Ubuntu 22.04 and 24.04
- On Windows, use WSL

## Install / Build

This is not an official Go library (yet) and must be cloned manually.

The code in this repository relies on the following submodule:
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

### Clone repo

```
git clone --recurse-submodules https://github.com/kylerohn/gollama.git
```

#### OR

```
git clone https://github.com/kylerohn/gollama.git
git submodule update --init --recursive
```

### Build backend

gollama requires llama.cpp to be built before use. This is handled via `go generate`:

```
go generate ./...
```

This step builds the llama.cpp backend and must be re-run if backend-related environment variables change.

By default, gollama is built for CPU execution using CMake with Makefiles. The following environment variables can be set to customize the build:

**Generators**
```
LLAMA_GENERATOR=Ninja
```

**Backends**
```
LLAMA_BACKEND=cuda
```

### Run example & test build

You can run the example under `examples/simple_chat` to verify that the build process completed successfully and to see how gollama works. Make sure you have followed the steps under [Build backend](#build-backend) before proceeding.

#### Install GGUF model

Download a GGUF model from Hugging Face or another source.  
For example: https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF

#### Edit model path

Update the model path in `examples/simple_chat/simple_chat.go`, replacing `path/to/gguf/model` with the actual path where the model is stored.

#### Build & run

> [!IMPORTANT]
> You must build before running. Using `go run` at this stage will not work.

Build the example:
```
go build ./examples/simple_chat
```

Run it:
```
./simple_chat
```

If successful, this will allow you to interact with the downloaded model.

### Configuration

The `simple_chat` example demonstrates a minimal configuration, primarily the `NumCtx` parameter for defining the context window size.

At present, configuration options closely mirror those provided by llama.cpp rather than abstracting them away. Until formal documentation is added, available settings can be found in `gollama/core/llama.go`, which maps directly to many native llama.cpp parameters.