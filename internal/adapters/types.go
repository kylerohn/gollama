package adapters

/*
#cgo CFLAGS: -I${SRCDIR}/../../external/llama.cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/bin -lllama -Wl,-rpath,$ORIGIN/build/bin

#include <stdlib.h>
#include <stdio.h>
#include <llama.h>
*/
import "C"

type AdapterLoRA = C.struct_llama_adapter_lora

type Batch = C.llama_batch

type ChatMessage = C.llama_chat_message

type Context = C.struct_llama_context
type ContextParams = C.struct_llama_context_params

type GGMLNumaStrategy = C.enum_ggml_numa_strategy

type LogitBias = C.llama_logit_bias

type MemoryT = C.llama_memory_t

type Model = C.struct_llama_model
type ModelParams = C.struct_llama_model_params

type ModelQuantizeParams = C.llama_model_quantize_params

type PerfContextData = C.struct_llama_perf_context_data
type PerfSamplerData = C.struct_perf_sampler_data

type PoolingT = C.enum_llama_pooling_type

type RopeT = C.enum_llama_rope_type

type Sampler = C.struct_llama_sampler
type SamplerI = C.struct_llama_sampler_i
type SamplerChainParams = C.struct_llama_sampler_chain_params
type SamplerContextT = C.llama_sampler_context_t

type TokenAttr = C.enum_llama_token_attr
type TokenDataArray = C.llama_token_data_array
type TokenT = C.llama_token

type Vocab = C.struct_llama_vocab
type VocabT = C.enum_llama_vocab_type