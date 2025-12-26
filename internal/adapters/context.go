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
	"fmt"
	"log"
	"runtime"
	"unsafe"
)



func Free(ctx *Context) {
	C.llama_free(ctx)
}

func NumCtx(ctx *Context) uint32 {
	return uint32(C.llama_n_ctx(ctx))
}

func NumBatch(ctx *Context) uint32 {
	return uint32(C.llama_n_batch(ctx))
}

func NumUBatch(ctx *Context) uint32 {
	return uint32(C.llama_n_ubatch(ctx))
}

func NumSeqMax(ctx *Context) uint32 {
	return uint32(C.llama_n_seq_max(ctx))
}

func GetModel(ctx *Context) *Model {
	return C.llama_get_model(ctx)
}

func GetMemory(ctx *Context) MemoryT {
	return C.llama_get_memory(ctx)
}

func PoolingType(ctx *Context) PoolingT {
	return C.llama_pooling_type(ctx)
}

func VocabType(vocab *Vocab) VocabT {
	return C.llama_vocab_type(vocab)
}

func VocabNumTokens(vocab *Vocab) int32 {
	return int32(C.llama_vocab_n_tokens(vocab))
}


// The following functions operate on a llama_context, hence the naming: llama_verb_...

// Add a loaded LoRA adapter to given context
// This will not modify model's weight
func SetAdapterLoRA(ctx *Context, adapter *AdapterLoRA, scale float32) int32 {
	return int32(C.llama_set_adapter_lora(ctx, adapter, C.float(scale)))
}

// Remove a specific LoRA adapter from given context
// Return -1 if the adapter is not present in the context
func RemoveAdapterLoRA(ctx *Context, adapter *AdapterLoRA) {
	res := C.llama_rm_adapter_lora(ctx, adapter)
	if res < 0 {
		log.Fatalf("RemoveAdapterLoRA: adapter at %p not present in context at %p", adapter, ctx)
	}
}

// Remove all LoRA adapters from given context
func ClearAdapterLoRA(ctx *Context) {
	C.llama_clear_adapter_lora(ctx)
}

// Apply a loaded control vector to a Context, or if data is empty, clear
// the currently loaded vector.
// nEmbed should be the size of a single layer's control, and data should point
// to an nEmbed x nLayers buffer starting from layer 1.
// ilStart and ilEnd are the layer range the vector should apply to (both inclusive)
// See llama_control_vector_load in common to load a control vector.
func ApplyAdapterCVec(ctx *Context, data []float32, nEmbed int32, ilStart int32, ilEnd int32) {

	errMsg := "ApplyAdapterCVec: failed to %s loaded control vector\nContext: %p\n%s"
	var res int32

	if len(data) < 1 {
		res = int32(C.llama_apply_adapter_cvec(ctx, nil, 0, C.int32_t(nEmbed), C.int32_t(ilStart), C.int32_t(ilEnd)))
		errMsg = fmt.Sprintf(errMsg, "clear", ctx, "")

	} else {
		res = int32(C.llama_apply_adapter_cvec(ctx, (*C.float)(unsafe.Pointer(&data[0])), C.size_t(len(data)), C.int32_t(nEmbed), C.int32_t(ilStart), C.int32_t(ilEnd)))
		errMsg = fmt.Sprintf(errMsg, "apply", ctx, fmt.Sprintf("Vector Length: %d\n", len(data)))
	}

	if res < 0 {
		errMsg = errMsg + "nEmbed: %d\nilStart: %d\nilEnd: %d\n"
		log.Fatalf(errMsg, nEmbed, ilStart, ilEnd)
	}
	runtime.KeepAlive(data)
}


/*
State/sessions
*/

// Returns the *actual* size in bytes of the state
// (logits, embedding and memory)
// Only use when saving the state, not when restoring it, otherwise the size may be too small.
func GetStateSize(ctx *Context) uint {
	return uint(C.llama_state_get_size(ctx))
}

// Copies the state to the specified destination address.
// Destination needs to have allocated enough memory.
func CopyStateData(ctx *Context) []byte {
	stateSize := GetStateSize(ctx)
	if stateSize == 0 {
		log.Fatalf("CopyStateData: State Size is zero\n")
	}

	buf := make([]byte, stateSize)
	copied := C.llama_state_get_data(ctx, (*C.uint8_t)(unsafe.Pointer(&buf[0])), C.size_t(stateSize))

	if copied == 0 {
		log.Fatalf("CopyStateData: 0 bytes written to buffer\n")
	}
	if uint(copied) > stateSize {
		log.Fatalf("CopyStateData: State larger than precomputed size")
	}
	runtime.KeepAlive(buf)
	return buf[:copied]
}

// Set the state reading from the specified address
// Returns the number of bytes read
func SetStateData(ctx *Context, src []byte) {
	stateSize := len(src)
	if stateSize == 0 {
		log.Fatalf("SetStateData: State Size is zero\n")
	}

	copied := C.llama_state_set_data(ctx, (*C.uint8_t)(unsafe.Pointer(&src[0])), C.size_t(stateSize))

	if copied == 0 {
		log.Fatalf("SetStateData: 0 bytes written to buffer\n")
	}
	runtime.KeepAlive(src)
}

// Save/load session file
func LoadStateFromFile(ctx *Context, statePath string) []TokenT {

	cPath := C.CString(statePath)
	defer C.free(unsafe.Pointer(cPath))

	max := NumCtx(ctx)

	if max < 1 {
		log.Fatalf("LoadStateFromFile: Context size must be greater than zero\n")
	}

	buf := make([]TokenT, max)
	var count C.size_t

	ok := C.llama_state_load_file(ctx, cPath, (*TokenT)(unsafe.Pointer(&buf[0])), C.size_t(max), (*C.size_t)(&count))

	if !ok {
		log.Fatalf("LoadStateFromFile: error loading from file %s\n", statePath)
	}

	runtime.KeepAlive(buf)

	n := min(int(count), len(buf))
	return buf[:n]

}

func SaveStateToFile(ctx *Context, statePath string, tokens []TokenT) {
	cPath := C.CString(statePath)
	defer C.free(unsafe.Pointer(cPath))

	ok := C.llama_state_save_file(ctx, cPath, (*TokenT)(unsafe.Pointer(&tokens[0])), C.size_t(len(tokens)))

	if !ok {
		log.Fatalf("SaveStateToFile: Cannot save state to %s", statePath)
	}
	runtime.KeepAlive(tokens)
}

// Get the exact size needed to copy the state of a single sequence
func GetSeqStateSize(ctx *Context, seq_id int32) uint {
	return uint(C.llama_state_seq_get_size(ctx, C.llama_seq_id(seq_id)))
}

// Copy the state of a single sequence into the specified buffer
func GetSeqStateData(ctx *Context, seq_id int32) []uint8 {

	size := GetSeqStateSize(ctx, seq_id)
	buf := make([]uint8, size)

	ok := C.llama_state_seq_get_data(ctx, (*C.uint8_t)(unsafe.Pointer(&buf[0])), C.size_t(size), C.llama_seq_id(seq_id))
	if ok < 1 {
		log.Fatalf("GetSeqStateData: Cannot get data at sequence number %d\n", seq_id)
	}
	runtime.KeepAlive((buf))
	return buf

}

// Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
// Returns:
//   - Positive: Ok
//   - Zero: Failed to load
func SetSeqStateData(ctx *Context, src []uint8, dest_seq_id int32) {
	if len(src) < 1 {
		log.Fatalf("SetSeqStateData: length of sequence data must be greater than 0\n")
	}

	ok := C.llama_state_seq_set_data(ctx, (*C.uint8_t)(unsafe.Pointer(&src[0])), C.size_t(len(src)), C.llama_seq_id(dest_seq_id))

	if ok < 1 {
		log.Fatalf("SetSeqStateData: cannot set sequence state to destination %d\n", dest_seq_id)
	}
	runtime.KeepAlive(src)
}

func SaveSeqStateToFile(ctx *Context, statePath string, seq_id int32, tokens []TokenT) {
	if len(tokens) < 1 {
		log.Fatalf("SaveSeqStateToFile: number of tokens must be greater than 0\n")
	}
	cPath := C.CString(statePath)
	defer C.free(unsafe.Pointer(cPath))

	ok := C.llama_state_seq_save_file(ctx, cPath, C.llama_seq_id(seq_id), (*TokenT)(unsafe.Pointer(&tokens[0])), C.size_t(len(tokens)))

	if ok < 1 {
		log.Fatalf("SaveSeqStateToFile: Cannot save state to %s", statePath)
	}
	runtime.KeepAlive(tokens)
}

func LoadSeqStateFromFile(ctx *Context, statePath string, dest_seq_id int32) []TokenT {
	cPath := C.CString(statePath)
	defer C.free(unsafe.Pointer(cPath))

	max := NumCtx(ctx)

	if max < 1 {
		log.Fatalf("LoadStateFromFile: Context size must be greater than zero\n")
	}

	buf := make([]TokenT, max)
	var count C.size_t

	ok := C.llama_state_seq_load_file(ctx, cPath, C.llama_seq_id(dest_seq_id), (*TokenT)(unsafe.Pointer(&buf[0])), C.size_t(max), (*C.size_t)(&count))

	if ok < 1 {
		log.Fatalf("LoadStateFromFile: error loading from file %s\n", statePath)
	}
	runtime.KeepAlive(buf)

	n := min(int(count), len(buf))
	return buf[:n]

}

// work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
func GetSeqStateSizeExt(ctx *Context, seq_id int32, flags uint32) uint {
	return uint(C.llama_state_seq_get_size_ext(ctx, C.llama_seq_id(seq_id), C.llama_state_seq_flags(flags)))
}

func GetSeqStateDataExt(ctx *Context, seq_id int32, flags uint32) []uint8 {

	size := GetSeqStateSizeExt(ctx, seq_id, flags)
	buf := make([]uint8, size)

	ok := C.llama_state_seq_get_data_ext(ctx, (*C.uint8_t)(unsafe.Pointer(&buf[0])), C.size_t(size), C.llama_seq_id(seq_id), C.llama_state_seq_flags(flags))
	if ok < 1 {
		log.Fatalf("GetSeqStateDataExt: Cannot get data at sequence number %d\n", seq_id)
	}
	runtime.KeepAlive((buf))
	return buf
}

func SetSeqStateDataExt(ctx *Context, src []uint8, dest_seq_id int32, flags uint32) {
	if len(src) < 1 {
		log.Fatalf("SetSeqStateData: length of sequence data must be greater than 0\n")
	}

	ok := C.llama_state_seq_set_data_ext(ctx, (*C.uint8_t)(unsafe.Pointer(&src[0])), C.size_t(len(src)), C.llama_seq_id(dest_seq_id), C.llama_state_seq_flags(flags))

	if ok < 1 {
		log.Fatalf("SetSeqStateDataExt: cannot set sequence state to destination %d\n", dest_seq_id)
	}
	runtime.KeepAlive(src)
}


// Process a batch of tokens.
// In contrast to llama_decode() - this call does not use KV cache.
// For encode-decoder contexts, processes the batch using the encoder.
// Can store the encoder output internally for later use by the decoder's cross-attention layers.
//
//	0 - success
//
// < 0 - error. the memory state is restored to the state before this call
func EncodeBatch(ctx *Context, batch Batch) {
	ok := C.llama_encode(ctx, batch)
	if ok < 0 {
		log.Fatalf("EncodeBatch: error encoding batch\n")
	}
}

// Process a batch of tokens.
// Requires the context to have a memory.
// For encode-decoder contexts, processes the batch using the decoder.
// Positive return values does not mean a fatal error, but rather a warning.
// Upon fatal-error or abort, the ubatches that managed to be been processed will remain in the memory state of the context
//
//	To handle this correctly, query the memory state using llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
//
// Upon other return values, the memory state is restored to the state before this call
//
//	 0 - success
//	 1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
//	 2 - aborted     (processed ubatches will remain in the context's memory)
//	-1 - invalid input batch
//
// < -1 - fatal error (processed ubatches will remain in the context's memory)
func DecodeBatch(ctx *Context, batch Batch) {
	ok := C.llama_decode(ctx, batch)
	// TODO unsure these should be fatal
	if ok == 1 {
		log.Fatalf("DecodeBatch: could not find a KV slot for the batch (try reducing the size of the batch or increase the context)\n")
	} else if ok == 2 {
		log.Fatalf("DecodeBatch: aborted (processed ubatches will remain in the context's memory)\n")
	} else if ok == -1 {
		log.Fatalf("DecodeBatch: invalid input batch\n")
	} else if ok < -1 {
		log.Fatalf("DecodeBatch: fatal error (processed ubatches will remain in the context's memory)\n")
	}
}

// Set the number of threads used for decoding
// n_threads is the number of threads used for generation (single token)
// n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
func SetNumDecodingThreads(ctx *Context, n_threads int32, n_threads_batch int32) {
	C.llama_set_n_threads(ctx, C.int32_t(n_threads), C.int32_t(n_threads_batch))
}

// Get the number of threads used for generation of a single token.
func GetNumDecodingThreads(ctx *Context) int32 {
	return int32(C.llama_n_threads(ctx))
}

// Get the number of threads used for prompt and batch processing (multiple token).
func GetNumDecodingThreadsBatch(ctx *Context) int32 {
	return int32(C.llama_n_threads_batch(ctx))
}

// Set whether the context outputs embeddings or not
// from llama.cpp -> TODO: rename to avoid confusion with llama_get_embeddings()
func SetEmbeddings(ctx *Context, embeddings bool) {
	C.llama_set_embeddings(ctx, C.bool(embeddings))
}

// Set whether to use causal attention or not
// If set to true, the model will only attend to the past tokens
func SetCausalAttn(ctx *Context, casual_attn bool) {
	C.llama_set_causal_attn(ctx, C.bool(casual_attn))
}

// Set whether the model is in warmup mode or not
// If true, all model tensors are activated during llama_decode() to load and cache their weights.
func SetWarmup(ctx *Context, warmup bool) {
	C.llama_set_warmup(ctx, C.bool(warmup))
}

// Wait until all computations are finished
// This is automatically done when using one of the functions below to obtain the computation results
// and is not necessary to call it explicitly in most cases
func Synchronize(ctx *Context) {
	C.llama_synchronize(ctx)
}

// Logits for the ith token. For positive indices, Equivalent to:
// llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
// Negative indicies can be used to access logits in reverse order, -1 is the last logit.
// returns NULL for invalid ids.
// TODO can i get vocab from context, or just pass in the nTokens directly?
func GetIthTokenLogits(ctx *Context, vocab *Vocab, i int32) []float32 {

	nTokens := VocabNumTokens(vocab)
	if nTokens <= 0 {
		log.Fatalf("GetIthTokenLogits: invalid number of tokens %d\n", nTokens)
	}

	ptr := C.llama_get_logits_ith(ctx, C.int32_t(i))
	if ptr == nil {
		log.Fatalf("GetIthTokenLogits: invalid index %d\n", i)
	}

	view := unsafe.Slice((*C.float)(unsafe.Pointer(ptr)), nTokens)

	out := make([]float32, nTokens)
	for j := range nTokens {
		out[j] = float32(view[j])
	}

	return out
}



// Get the embeddings for the ith token. For positive indices, Equivalent to:
// llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
// Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
// shape: [n_embd] (1-dimensional)
// returns NULL for invalid ids.
func GetIthEmbeddings(ctx *Context, model *Model, i int32) []float32 {
	nEmbd := ModelNumEmbd(model)
	if nEmbd <= 0 {
		log.Fatalf("GetIthEmbeddings: invalid embedding size %d\n", nEmbd)
	}

	ptr := C.llama_get_embeddings_ith(ctx, C.int32_t(i))
	if ptr == nil {
		log.Fatalf("GetEmbeddingsIth: invalid index %d\n", i)
	}

	view := unsafe.Slice((*C.float)(unsafe.Pointer(ptr)), nEmbd)

	out := make([]float32, nEmbd)
	for j := range nEmbd {
		out[j] = float32(view[j])
	}

	return out
}

// TODO same for this as GetIthTokenLogits
func GetIthSeqEmbeddings(ctx *Context, model *Model, seq_id int32) []float32 {
	nEmbd := ModelNumEmbd(model)
	if nEmbd <= 0 {
		log.Fatalf("GetIthSeqEmbeddings: invalid embedding size %d\n", nEmbd)
	}

	ptr := C.llama_get_embeddings_seq(ctx, C.llama_seq_id(seq_id))
	if ptr == nil {
		log.Fatalf("GetIthSeqEmbeddings: invalid sequence id %d\n", seq_id)
	}

	view := unsafe.Slice((*C.float)(unsafe.Pointer(ptr)), nEmbd)

	out := make([]float32, nEmbd)
	for j := range nEmbd {
		out[j] = float32(view[j])
	}

	return out
}


// Set abort callback
// TODO: how tf do I do this
// LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);