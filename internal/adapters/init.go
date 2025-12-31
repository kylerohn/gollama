package adapters

/*
#cgo CFLAGS: -I${SRCDIR}/../../external/llama.cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/bin -lllama -Wl,-rpath,$ORIGIN/build/bin

#include <stdlib.h>
#include <stdio.h>
#include <llama.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

/*
Helper fn wrappers for default parameters
*/
func ModelDefaultParams() ModelParams {
	return C.llama_model_default_params()
}

func ContextDefaultParams() ContextParams {
	return C.llama_context_default_params()
}

func SamplerChainDefaultParams() SamplerChainParams {
	return C.llama_sampler_chain_default_params()
}

func ModelQuantizeDefaultParams() ModelQuantizeParams {
	return C.llama_model_quantize_default_params()
}

func BackendInit() {
	C.llama_backend_init()
}

// Currently only used for MPI
func BackendFree() {
	C.llama_backend_free()
}

// Optional
func NumaInit(numa GGMLNumaStrategy) {
	C.llama_numa_init(numa)
}

func LoadModelFromFile(modelPath string, params ModelParams) *Model {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	return C.llama_model_load_from_file(cPath, params)
}

func LoadModelFromSplits(paths []string, params ModelParams) *Model {

	nPaths := uint(len(paths))
	cPaths := make([]*C.char, 0, nPaths)

	for _, path := range paths {
		cStr := C.CString(path)
		cPaths = append(cPaths, cStr)
		defer C.free(unsafe.Pointer(cStr))
	}

	res := C.llama_model_load_from_splits((**C.char)(unsafe.Pointer(&cPaths[0])), C.size_t(nPaths), params)
	runtime.KeepAlive(cPaths)
	return res
}

func SaveModelToFile(model *Model, modelPath string) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	C.llama_model_save_to_file(model, cPath)
}

func InitFromModel(model *Model, params ContextParams) *Context {
	return C.llama_init_from_model(model, params)
}
