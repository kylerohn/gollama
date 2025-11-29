package adapters

/*
#cgo CFLAGS: -I${SRCDIR}/../../external/llama.cpp/include
#cgo CFLAGS: -I${SRCDIR}/../../external/llama.cpp/ggml/include
#cgo CFLAGS: -I${SRCDIR}/../../shims/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/bin -lggml -lllama -lshims -Wl,-rpath,$ORIGIN/build/bin

#include <stdlib.h>
#include <stdio.h>
#include <ggml.h>
#include <llama.h>
#include <shims.h>
*/
import "C"
import (
	"fmt"
	"log"
	"runtime"
	"unsafe"
)

/*
Type Aliases for c -> go
*/

type AdapterLoRA = C.struct_llama_adapter_lora

type Context = C.struct_llama_context
type ContextParams = C.struct_llama_context_params

type GGMLNumaStrategy = C.enum_ggml_numa_strategy

type MemoryT = C.llama_memory_t

type Model = C.struct_llama_model
type ModelParams = C.struct_llama_model_params

type ModelQuantizeParams = C.llama_model_quantize_params

type PoolingT = C.enum_llama_pooling_type

type RopeT = C.enum_llama_rope_type

type Sampler = C.struct_llama_sampler
type SamplerChainParams = C.struct_llama_sampler_chain_params

type TokenT = C.int32_t

type Vocab = C.struct_llama_vocab
type VocabT = C.enum_llama_vocab_type

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

/*
Program Initialization/Conclusion
*/

func GGMLBackendLoadAll() {
	C.ggml_backend_load_all()
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

func FreeModel(model *Model) {
	C.llama_model_free(model)
}

func InitFromModel(model *Model, params ContextParams) *Context {
	return C.llama_init_from_model(model, params)
}

func Free(ctx *Context) {
	C.llama_free(ctx)
}

/*
Program Information
*/
func TimeUs() int64 {
	return int64(C.llama_time_us())
}

func MaxDevices() uint {
	return uint(C.llama_max_devices())
}

func MaxParallelSequence() uint {
	return uint(C.llama_max_parallel_sequences())
}

func SupportsMmap() bool {
	return bool(C.llama_supports_mmap())
}

func SupportsMlock() bool {
	return bool(C.llama_supports_mlock())
}

func SupportsGPUOffload() bool {
	return bool(C.llama_supports_gpu_offload())
}

func SupportsRPC() bool {
	return bool(C.llama_supports_rpc())
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

func GetModelVocab(model *Model) *Vocab {
	return C.llama_model_get_vocab(model)
}

func ModelRopeType(model *Model) RopeT {
	return C.llama_model_rope_type(model)
}

func ModelNumCtxTrain(model *Model) int32 {
	return int32(C.llama_model_n_ctx_train(model))
}

func ModelNumEmbd(model *Model) int32 {
	return int32(C.llama_model_n_embd(model))
}

func ModelNumLayer(model *Model) int32 {
	return int32(C.llama_model_n_layer(model))
}

func ModelNumHead(model *Model) int32 {
	return int32(C.llama_model_n_head(model))
}

func ModelNumHeadKV(model *Model) int32 {
	return int32(C.llama_model_n_head_kv(model))
}

func ModelNumSwa(model *Model) int32 {
	return int32(C.llama_model_n_swa(model))
}

// Model's RoPE frequency scaling factor
func ModelRopeFreqScaleTrain(model *Model) float32 {
	return float32(C.llama_model_rope_freq_scale_train(model))
}

// Returns number of classifier outputs for classifier models, else undefined behavior
func ModelNumClsOut(model *Model) uint32 {
	return uint32(C.llama_model_n_cls_out(model))
}

// Returns label of classifier outbut by index (<NumClsOut), or nil if no label provided
func ModelClsLabel(model *Model, i uint32) string {
	res := C.llama_model_cls_label(model, C.uint(i))
	if res == nil {
		return ""
	}
	return C.GoString(res)
}

func VocabType(vocab *Vocab) VocabT {
	return C.llama_vocab_type(vocab)
}

func VocabNumTokens(vocab *Vocab) int32 {
	return int32(C.llama_vocab_n_tokens(vocab))
}

/*
GGUF Metadata scalar values
*/

// Metadata value as a string by key name
func ModelMetaValStr(model *Model, key string) string {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	// TODO Add Bounded Growth
	var size uint = 16384
	buf := make([]byte, size)

	strLen := int32(C.llama_model_meta_val_str(model, cKey, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(size)))

	if strLen < 0 {
		log.Fatalf("ModelMetaValStr: failed to retrieve key %s\n", key)
	}
	if uint(strLen) < size-1 {
		res := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
		runtime.KeepAlive(buf)
		return res
	}

	log.Fatalf("ModelMetaValStr: result length too large for buffer (%d > %d)\n", strLen, size)
	return ""
}

// get number of metadata key/value pairs
func ModelMetaCount(model *Model) int32 {
	return int32(C.llama_model_meta_count(model))
}

// get metadata key name by index
func ModelMetaKeyByIndex(model *Model, i int) string {
	// TODO Add Bounded Growth
	var size uint = 16384
	buf := make([]byte, size)

	strLen := int32(C.llama_model_meta_key_by_index(model, C.int(i), (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(size)))

	if strLen < 0 {
		log.Fatalf("ModelMetaKeyByIndex: failed to retrieve key at index %d\n", i)
	}
	if uint(strLen) < size-1 {
		res := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
		runtime.KeepAlive(buf)
		return res
	}

	log.Fatalf("ModelMetaKeyByIndex: result length too large for buffer (%d > %d)\n", strLen, size)
	return ""
}

// get metadata key value by index
func ModelMetaValByIndex(model *Model, i int) string {
	// TODO Add Bounded Growth
	var size uint = 16384
	buf := make([]byte, size)

	strLen := int32(C.llama_model_meta_val_str_by_index(model, C.int(i), (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(size)))

	if strLen < 0 {
		log.Fatalf("ModelMetaValByIndex: failed to retrieve value at index %d\n", i)
	}
	if uint(strLen) < size-1 {
		res := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
		runtime.KeepAlive(buf)
		return res
	}

	log.Fatalf("ModelMetaValByIndex: result length too large for buffer (%d > %d)\n", strLen, size)
	return ""
}

// Get a string describing the model type
func ModelDesc(model *Model) string {
	// TODO Add Bounded Growth
	var size uint = 16384
	buf := make([]byte, size)

	strLen := int32(C.llama_model_desc(model, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(size)))

	if strLen < 0 {
		log.Fatalf("ModelDesc: failed to retrieve model description metadata \n")
	}
	if uint(strLen) < size-1 {
		res := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
		runtime.KeepAlive(buf)
		return res
	}

	log.Fatalf("ModelDesc: result length too large for buffer (%d > %d)\n", strLen, size)
	return ""
}

// Returns the total size of all the tensors in the model in bytes
func ModelSize(model *Model) uint64 {
	return uint64(C.llama_model_size(model))
}

// Get the default chat template. Returns nullptr if not available
// If name is "", returns the default chat template
func ModelChatTemplate(model *Model, name string) string {

	var res *C.char

	if name != "" {
		cStr := C.CString(name)
		defer C.free(unsafe.Pointer(cStr))
		res = C.llama_model_chat_template(model, cStr)

	} else {
		res = C.llama_model_chat_template(model, nil)
	}
	defer C.free(unsafe.Pointer(res))

	if res == nil {
		if name == "" {
			name = "[DEFAULT]"
		}
		log.Fatalf("ModelChatTemplate: could not find chat template for name %s\n", name)
	}

	return C.GoString(res)
}

// Returns the total number of parameters in the model
func ModelNumParams(model *Model) uint64 {
	return uint64(C.llama_model_n_params(model))
}

// Returns true if the model contains an encoder that requires llama_encode() call
func ModelHasEncoder(model *Model) bool {
	return bool(C.llama_model_has_encoder(model))
}

// Returns true if the model contains a decoder that requires llama_decode() call
func ModelHasDecoder(model *Model) bool {
	return bool(C.llama_model_has_decoder(model))
}

// For encoder-decoder models, this function returns id of the token that must be provided
// to the decoder to start generating output sequence. For other models, it returns -1.
func ModelDecoderStartToken(model *Model) TokenT {
	return C.llama_model_decoder_start_token(model)
}

// Returns true if the model is recurrent (like Mamba, RWKV, etc.)
func ModelIsRecurrent(model *Model) bool {
	return bool(C.llama_model_is_recurrent(model))
}

// Returns true if the model is hybrid (like Jamba, Granite, etc.)
func ModelIsHybrid(model *Model) bool {
	return bool(C.llama_model_is_hybrid(model))
}

// Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
func ModelIsDiffusion(model *Model) bool {
	return bool(C.llama_model_is_diffusion(model))
}

// Returns 0 on success
func ModelQuantize(fnameIn string, fnameOut string, params *ModelQuantizeParams) {
	cStrIn := C.CString(fnameIn)
	cStrOut := C.CString(fnameOut)
	defer C.free(unsafe.Pointer(cStrIn))
	defer C.free(unsafe.Pointer(cStrOut))

	failure := uint32(C.llama_model_quantize(cStrIn, cStrOut, params))

	if failure != 0 {
		log.Fatalf("Could not Quantize Model at %s to %s\n", fnameIn, fnameOut)
	}

}

/*
Adapters
*/

// Load a LoRA adapter from file
func InitAdapterLoRA(model *Model, loraPath string) *AdapterLoRA {
	cPath := C.CString(loraPath)
	defer C.free(unsafe.Pointer(cPath))

	adapter := C.llama_adapter_lora_init(model, cPath)
	if adapter == nil {
		log.Fatalf("InitAdapterLoRA: failed Loading LoRA adapter from path %s\n", loraPath)
	}
	return adapter
}

/*
Functions to access the adapter's GGUF metadata scalar values
*/

// Metadata value as a string by key name
func AdapterMetaValStr(adapter *AdapterLoRA, key string) string {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	// TODO Add Bounded Growth
	var size uint = 16384
	buf := make([]byte, size)

	strLen := int32(C.llama_adapter_meta_val_str(adapter, cKey, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(size)))

	if strLen < 0 {
		log.Fatalf("AdapterMetaValStr: failed to retrieve key %s\n", key)
	}
	if uint(strLen) < size-1 {
		res := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
		runtime.KeepAlive(buf)
		return res
	}

	log.Fatalf("AdapterMetaValStr: result length too large for buffer (%d > %d)\n", strLen, size)
	return ""
}

// get number of metadata key/value pairs
func AdapterMetaCount(adapter *AdapterLoRA) int32 {
	return int32(C.llama_adapter_meta_count(adapter))
}

// get metadata key name by index
func AdapterMetaKeyByIndex(adapter *AdapterLoRA, i int) string {
	// TODO Add Bounded Growth
	var size uint32 = 16384
	buf := make([]byte, size)

	strLen := int32(C.llama_adapter_meta_key_by_index(adapter, C.int(i), (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(size)))

	if strLen < 0 {
		log.Fatalf("AdapterMetaKeyByIndex: failed to retrieve key at index %d\n", i)
	}
	if uint32(strLen) < size-1 {
		res := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
		runtime.KeepAlive(buf)
		return res
	}

	log.Fatalf("AdapterMetaKeyByIndex: result length too large for buffer (%d > %d)\n", strLen, size)
	return ""
}

// get metadata key value by index
func AdapterMetaValByIndex(adapter *AdapterLoRA, i int) string {
	// TODO Add Bounded Growth
	var size uint = 16384
	buf := make([]byte, size)

	strLen := int32(C.llama_adapter_meta_val_str_by_index(adapter, C.int(i), (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(size)))

	if strLen < 0 {
		log.Fatalf("AdapterMetaValByIndex: failed to retrieve value at index %d\n", i)
	}
	if uint(strLen) < size-1 {
		res := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
		runtime.KeepAlive(buf)
		return res
	}

	log.Fatalf("AdapterMetaValByIndex: result length too large for buffer (%d > %d)\n", strLen, size)
	return ""
}

// Manually free a LoRA adapter
// Note: loaded adapters will be free when the associated model is deleted
func AdapterLoRAFree(adapter *AdapterLoRA) {
	C.llama_adapter_lora_free(adapter)
}

// Get the invocation tokens if the current lora is an alora
func AdapterGetALoRANumInvocationTokens(adapter *AdapterLoRA) uint64 {
	return uint64(C.llama_adapter_get_alora_n_invocation_tokens(adapter))
}
func AdapterGetALoRAInvocationTokens(adapter *AdapterLoRA) []TokenT {
	tokenPtr := C.llama_adapter_get_alora_invocation_tokens(adapter)
	if tokenPtr == nil {
		log.Fatalf("AdapterGetLoRAInvocationTokens: internal c method returned nullptr\n")
	}
	count := AdapterGetALoRANumInvocationTokens(adapter)
	res := unsafe.Slice((*TokenT)(tokenPtr), count)
	return res
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
Memory
*/

// Clear the memory contents
// If data == true, the data buffers will also be cleared together with the metadata
func ClearMemory(mem MemoryT, data bool) {
	C.llama_memory_clear(mem, C.bool(data))
}

// Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
// Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
// seq_id < 0 : match any sequence
// p0 < 0     : [0,  p1]
// p1 < 0     : [p0, inf)
func RemoveMemorySequence(mem MemoryT, seqId int32, p0 int32, p1 int32) {
	if !C.llama_memory_seq_rm(mem, C.int32_t(seqId), C.int32_t(p0), C.int32_t(p1)) {
		log.Fatalf("RemoveMemorySequence: position %d to %d in sequence %d cannot be removed\n", p0, p1, seqId)
	}
}

// Copy all tokens that belong to the specified sequence to another sequence
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
func CopyMemorySequence(mem MemoryT, seqIdSrc int32, seqIdDst int32, p0 int32, p1 int32) {
	C.llama_memory_seq_cp(mem, C.int32_t(seqIdSrc), C.int32_t(seqIdDst), C.int32_t(p0), C.int32_t(p1))
}

// Removes all tokens that do not belong to the specified sequence
func KeepMemorySequence(mem MemoryT, seqId int32) {
	C.llama_memory_seq_keep(mem, C.int32_t(seqId))
}

// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
func AddMemorySequence(mem MemoryT, seqId int32, p0 int32, p1 int32, delta int32) {
	C.llama_memory_seq_add(mem, C.int32_t(seqId), C.int32_t(p0), C.int32_t(p1), C.int32_t(delta))
}

// Integer division of the positions by factor of `d > 1`
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
func DivideMemorySequence(mem MemoryT, seqId int32, p0 int32, p1 int32, d int) {
	C.llama_memory_seq_div(mem, C.int32_t(seqId), C.int32_t(p0), C.int32_t(p1), C.int(d))
}

// Returns the smallest position present in the memory for the specified sequence
// This is typically non-zero only for SWA caches
// Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
// Return -1 if the sequence is empty
func MinMemorySequence(mem MemoryT, seqId int32) int32 {
	return int32(C.llama_memory_seq_pos_min(mem, C.int32_t(seqId)))
}

// Returns the largest position present in the memory for the specified sequence
// Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
// Return -1 if the sequence is empty
func MaxMemorySequence(mem MemoryT, seqId int32) int32 {
	return int32(C.llama_memory_seq_pos_max(mem, C.int32_t(seqId)))
}

// Check if the memory supports shifting
func MemoryCanShift(mem MemoryT) bool {
	return bool(C.llama_memory_can_shift(mem))
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
// Returns the number of bytes copied
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

	ok := C.llama_state_load_file(ctx, cPath, (*TokenT)(unsafe.Pointer(&buf[0])), C.size_t(max), (*C.size_t)(unsafe.Pointer(&count)))

	if !ok {
		log.Fatalf("LoadStateFromFile: error loading from file %s\n", statePath)
	}

	if count > C.ulong(len(buf)) {
		count = C.ulong(len(buf))
	}

	runtime.KeepAlive(buf)

	return buf[:count]
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

/*

// Get the exact size needed to copy the state of a single sequence
LLAMA_API size_t llama_state_seq_get_size(
		struct llama_context * ctx,
				llama_seq_id   seq_id);

// Copy the state of a single sequence into the specified buffer
LLAMA_API size_t llama_state_seq_get_data(
		struct llama_context * ctx,
						uint8_t * dst,
						size_t   size,
				llama_seq_id   seq_id);

// Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
// Returns:
//  - Positive: Ok
//  - Zero: Failed to load
LLAMA_API size_t llama_state_seq_set_data(
		struct llama_context * ctx,
				const uint8_t * src,
						size_t   size,
				llama_seq_id   dest_seq_id);

LLAMA_API size_t llama_state_seq_save_file(
		struct llama_context * ctx,
					const char * filepath,
				llama_seq_id   seq_id,
			const llama_token * tokens,
						size_t   n_token_count);

LLAMA_API size_t llama_state_seq_load_file(
		struct llama_context * ctx,
					const char * filepath,
				llama_seq_id   dest_seq_id,
					llama_token * tokens_out,
						size_t   n_token_capacity,
						size_t * n_token_count_out);

typedef uint32_t llama_state_seq_flags;

LLAMA_API size_t llama_state_seq_get_size_ext(
		struct llama_context * ctx,
				llama_seq_id   seq_id,
		llama_state_seq_flags   flags);

LLAMA_API size_t llama_state_seq_get_data_ext(
		struct llama_context * ctx,
						uint8_t * dst,
						size_t   size,
				llama_seq_id   seq_id,
		llama_state_seq_flags   flags);

LLAMA_API size_t llama_state_seq_set_data_ext(
		struct llama_context * ctx,
				const uint8_t * src,
						size_t   size,
				llama_seq_id   dest_seq_id,
		llama_state_seq_flags   flags);

//
// Decoding
//

*/
