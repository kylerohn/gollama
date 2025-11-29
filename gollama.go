package gollama

/*
#cgo CFLAGS: -I${SRCDIR}/external/llama.cpp/include
#cgo CFLAGS: -I${SRCDIR}/external/llama.cpp/ggml/include
#cgo CFLAGS: -I${SRCDIR}/shims/include
#cgo LDFLAGS: -L${SRCDIR}/build/bin -lggml -lllama -lshims -Wl,-rpath,$ORIGIN/build/bin

#include <stdio.h>
#include <ggml.h>
#include <llama.h>
#include <shims.h>
*/
import "C"
import (
	"log"
)

func InitializeModel(filepath string, n_gpu_layers int, n_ctx int) Model {

	C.ggml_backend_load_all()

	model_params := C.llama_model_default_params()
	model_params.n_gpu_layers = C.int(n_gpu_layers)

	model := C.llama_model_load_from_file(C.CString(filepath), model_params)
	if model == nil {
		log.Fatal("Failed to load model")
	}

	vocab := C.llama_model_get_vocab(model)

	ctx_params := C.llama_context_default_params()
	ctx_params.n_ctx = C.uint(n_ctx)
	ctx_params.n_batch = C.uint(n_ctx)

	context := C.llama_init_from_model(model, ctx_params)

	sampler := C.llama_sampler_chain_init(C.llama_sampler_chain_default_params())
	C.llama_sampler_chain_add(sampler, C.llama_sampler_init_min_p(0.05, 1))
	C.llama_sampler_chain_add(sampler, C.llama_sampler_init_temp(0.8))
	C.llama_sampler_chain_add(sampler, C.llama_sampler_init_dist(C.LLAMA_DEFAULT_SEED))

	return Model{Model: model, Vocab: vocab, Context: context, Sampler: sampler}
}
