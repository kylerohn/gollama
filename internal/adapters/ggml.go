package adapters

/*
#cgo CFLAGS: -I${SRCDIR}/../../external/llama.cpp/include
#cgo CFLAGS: -I${SRCDIR}/../../external/llama.cpp/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/bin -lllama -lggml -Wl,-rpath,$ORIGIN/build/bin

#include <stdlib.h>
#include <stdio.h>
#include <ggml.h>
#include <llama.h>
*/
import "C"

func GGMLBackendLoadAll() {
	C.ggml_backend_load_all()
}
