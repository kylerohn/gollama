#!/usr/bin/bash

set -euo pipefail

BACKEND="${LLAMA_BACKEND:-cpu}"
GENERATOR="${LLAMA_GENERATOR:-'Unix Makefiles'}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="$ROOT/build"

echo "==> gollama native build"
echo "    backend   : $BACKEND"
echo "    generator : $GENERATOR"
echo "    build dir : $BUILD"
echo

if [[ -d "$BUILD" ]]; then
    echo "==> Removing existing build directory"
    rm -rf "$BUILD"
fi

echo "==> Configuring llama.cpp"

case "$BACKEND" in
    cpu)
        cmake -S "$ROOT" -B "$BUILD" -G "$GENERATOR" \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_EXAMPLES=OFF
        ;;
    cuda)
        cmake -S "$ROOT" -B "$BUILD" -G "$GENERATOR" \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DGGML_CUDA=ON
        ;;
    *)
        echo "!! Unknown backend: $BACKEND (expected cpu or cuda)"
        exit 2
        ;;
esac

echo "==> Building"
cmake --build "$BUILD" -j

echo
echo "==> Build complete"
echo "    backend : $BACKEND"
echo "    output  : $BUILD"