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

type Batch = C.llama_batch

type ChatMessage = C.llama_chat_message

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

type TokenAttr = C.enum_llama_token_attr
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

//
// Decoding
//

// Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
// Each token can be assigned up to n_seq_max sequence ids
// The batch has to be freed with llama_batch_free()
// If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
// Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
// The rest of the llama_batch members are allocated with size n_tokens
// All members are left uninitialized
func InitBatch(n_tokens int32, embd int32, n_seq_max int32) Batch {
	return C.llama_batch_init(C.int32_t(n_tokens), C.int32_t(embd), C.int32_t(n_seq_max))
}

// Frees a batch of tokens allocated with llama_batch_init()
func FreeBatch(batch Batch) {
	C.llama_batch_free(batch)
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

// Set abort callback
// TODO: how tf do I do this
// LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

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

//
// Vocab
//

func GetVocabText(vocab *Vocab, token TokenT) string {
	cStr := C.llama_vocab_get_text(vocab, token)
	out := C.GoString(cStr)
	return out
}

func GetVocabScore(vocab *Vocab, token TokenT) float32 {
	return float32(C.llama_vocab_get_score(vocab, token))
}

func GetVocabAttr(vocab *Vocab, token TokenT) TokenAttr {
	return C.llama_vocab_get_attr(vocab, token)
}

// Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
func VocabIsEOG(vocab *Vocab, token TokenT) bool {
	return bool(C.llama_vocab_is_eog(vocab, token))
}

// Identify if Token Id is a control token or a render-able token
func VocabIsControl(vocab *Vocab, token TokenT) bool {
	return bool(C.llama_vocab_is_control(vocab, token))
}

// Special tokens
// beginning-of-sentence
func VocabBOS(vocab *Vocab) TokenT {
	return C.llama_vocab_bos(vocab)
}

// end-of-sentence
func VocabEOS(vocab *Vocab) TokenT {
	return C.llama_vocab_eos(vocab)
}

// end-of-turn
func VocabEOT(vocab *Vocab) TokenT {
	return C.llama_vocab_eot(vocab)
}

// sentence separator
func VocabSep(vocab *Vocab) TokenT {
	return C.llama_vocab_sep(vocab)
}

// next-line
func VocabNL(vocab *Vocab) TokenT {
	return C.llama_vocab_nl(vocab)
}

// padding
func VocabPad(vocab *Vocab) TokenT {
	return C.llama_vocab_pad(vocab)
}

// mask
func VocabMask(vocab *Vocab) TokenT {
	return C.llama_vocab_mask(vocab)
}

func GetAddBOS(vocab *Vocab) bool {
	return bool(C.llama_vocab_get_add_bos(vocab))
}

func GetAddEOS(vocab *Vocab) bool {
	return bool(C.llama_vocab_get_add_eos(vocab))
}

func GetAddSep(vocab *Vocab) bool {
	return bool(C.llama_vocab_get_add_sep(vocab))
}

func VocabFimPre(vocab *Vocab) TokenT {
	return C.llama_vocab_fim_pre(vocab)
}

func VocabFimSuf(vocab *Vocab) TokenT {
	return C.llama_vocab_fim_suf(vocab)
}

func VocabFimMid(vocab *Vocab) TokenT {
	return C.llama_vocab_fim_mid(vocab)
}

func VocabFimPad(vocab *Vocab) TokenT {
	return C.llama_vocab_fim_pad(vocab)
}

func VocabFimRep(vocab *Vocab) TokenT {
	return C.llama_vocab_fim_rep(vocab)
}

func VocabFimSep(vocab *Vocab) TokenT {
	return C.llama_vocab_fim_sep(vocab)
}

// Tokenization
//
// The API is thread-safe.
//

// Tokenize converts text into tokens using vocab and returns up to nTokensMax tokens.
// It exits the program via log.Fatalf if nTokensMax < 1 or if tokenization fails.
//
// If addSpecial is true, BOS/EOS tokens may be added depending on the model
// configuration. If parseSpecial is true, special and control tokens are
// interpreted as tokens instead of plain text; parseSpecial does not insert
// a leading space.
func Tokenize(vocab *Vocab, text string, nTokensMax int32, addSpecial bool, parseSpecial bool) []TokenT {
	cStr := C.CString(text)
	defer C.free(unsafe.Pointer(cStr))

	if nTokensMax < 1 {
		log.Fatalf("Tokenize: max tokens is less than 1 (%d)\n", nTokensMax)
	}

	buf := make([]TokenT, nTokensMax)
	ok := C.llama_tokenize(vocab, cStr, C.int32_t(len(text)), (*TokenT)(unsafe.Pointer(&buf[0])), C.int32_t(nTokensMax), C.bool(addSpecial), C.bool(parseSpecial))

	if ok < 0 {
		log.Fatalf("Tokenize: llama_tokenize failed with code %d\n", ok)
	}
	if ok == 0 {
		return nil
	}
	n := int(ok)
	if n > len(buf) {
		log.Printf("Tokenize (WARN): Generated more tokens (%d) than buffer size (%d). Keeping only (%d) tokens\n", n, len(buf), len(buf))
		n = len(buf)
	}

	runtime.KeepAlive(buf)
	return buf[:n]
}

// Token Id -> Piece.
// Uses the vocabulary in the provided context.
// User can skip up to 'lStrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
// if special is true, special tokens are rendered in the output.
func TokenToPiece(vocab *Vocab, token TokenT, lstrip int32, special bool) string {
	bufSize := 128
	buf := make([]byte, bufSize)

	strLen := C.llama_token_to_piece(vocab, token, (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(bufSize), C.int32_t(lstrip), C.bool(special))

	if int(strLen) > len(buf) {
		log.Fatalf("TokenToPiece: length of output string (%d) greater than buffer (%d)\n", strLen, len(buf))
	}

	if strLen == 0 {
		return ""
	}

	out := C.GoStringN((*C.char)(unsafe.Pointer(&buf[0])), strLen)
	return out
}

// Detokenize converts a slice of tokens back into text (the inverse of tokenization).
// The destination buffer must be large enough to hold the full output.
//
// It returns the number of bytes written, up to textLenMax. On failure, it returns
// a negative value indicating how many bytes would have been required.
//
// If removeSpecial is true, BOS/EOS tokens may be removed depending on model
// configuration. If unparseSpecial is true, special tokens are rendered into
// the output rather than skipped.
func Detokenize(vocab *Vocab, tokens []TokenT, textLenMax int32, removeSpecial bool, unparseSpecial bool) string {
	if textLenMax < 1 {
		log.Fatalf("Detokenize: texLenMax is less than 1\n")
	}

	if len(tokens) == 0 {
		return ""
	}

	buf := make([]byte, textLenMax)
	strLen := C.llama_detokenize(vocab, (*TokenT)(unsafe.Pointer(&tokens[0])), C.int32_t(len(tokens)), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(textLenMax), C.bool(removeSpecial), C.bool(unparseSpecial))

	if strLen < 0 {
		log.Fatalf("Detokenize: buffer too small, need %d bytes (have %d)\n", -strLen, textLenMax)
	}

	if strLen == 0 {
		return ""
	}

	out := C.GoStringN((*C.char)(unsafe.Pointer(&buf[0])), strLen)

	runtime.KeepAlive(tokens)
	return out
}

//
// Chat templates
//

// ApplyChatTemplate formats a chat history using either the provided template
// or the model’s default template if tmpl is empty. It mirrors the behavior of
// HuggingFace’s apply_chat_template, but only supports the predefined template
// set used by llama_chat_apply_template.
//
// chat must contain at least one message. The function allocates a buffer large
// enough to hold the formatted prompt and grows it if needed. If addAssistant
// is true, the formatted output ends with the token(s) that begin an assistant
// reply.
//
// The returned string is the fully formatted prompt. The function terminates the
// program via log.Fatalf if template application fails.
func ApplyChatTemplate(tmpl string, chat []ChatMessage, addAssistant bool) string {
	if len(chat) < 1 {
		log.Fatalf("ApplyChatTemplate: No chat messages in input slice\n")
	}

	cStr := C.CString(tmpl)
	defer C.free(unsafe.Pointer(cStr))

	charCount := 0
	for _, message := range chat {
		charCount += len(C.GoString(message.content))
	}

	bufSize := max(charCount*2, 256)

	for {
		buf := make([]byte, bufSize)
		strLen := C.llama_chat_apply_template(cStr, (*ChatMessage)(unsafe.Pointer(&chat[0])), C.size_t(len(chat)), C.bool(addAssistant), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(bufSize))
		runtime.KeepAlive(chat)

		if strLen < 0 {
			log.Fatalf("ApplyChatTemplate: llama_chat_apply_template failed with code %d\n", strLen)
		}

		n := int(strLen)

		if strLen == 0 {
			return ""
		} else if n > bufSize {
			bufSize = n
		} else {
			return C.GoStringN((*C.char)(unsafe.Pointer(&buf[0])), strLen)
		}
	}
}

/*


    // Get list of built-in chat templates
    LLAMA_API int32_t llama_chat_builtin_templates(const char ** output, size_t len);

    //
    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = llama_sampler_chain_default_params();
    //
    //    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    //
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        llama_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
    //
    //        // accepting the token updates the internal state of certain samplers (e.g. grammar, repetition, etc.)
    //        llama_sampler_accept(smpl, id);
    //        ...
    //    }
    //
    //    llama_sampler_free(smpl);
    //
    // TODO: In the future, llama_sampler will be utilized to offload the sampling to the backends (e.g. GPU).
    //

    typedef void * llama_sampler_context_t;

    // user code can implement the interface below in order to create custom llama_sampler
    struct llama_sampler_i {
        const char *           (*name)  (const struct llama_sampler * smpl);                                 // can be NULL
        void                   (*accept)(      struct llama_sampler * smpl, llama_token token);              // can be NULL
        void                   (*apply) (      struct llama_sampler * smpl, llama_token_data_array * cur_p); // required
        void                   (*reset) (      struct llama_sampler * smpl);                                 // can be NULL
        struct llama_sampler * (*clone) (const struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL
        void                   (*free)  (      struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL

        // TODO: API for internal libllama usage for appending the sampling to an existing ggml_cgraph
        //void (*apply_ggml) (struct llama_sampler * smpl, ...);
    };

    struct llama_sampler {
        const struct llama_sampler_i * iface;
        llama_sampler_context_t        ctx;
    };

    // mirror of llama_sampler_i:
    LLAMA_API struct llama_sampler * llama_sampler_init  (const struct llama_sampler_i * iface, llama_sampler_context_t ctx);
    LLAMA_API const char *           llama_sampler_name  (const struct llama_sampler * smpl);
    LLAMA_API void                   llama_sampler_accept(      struct llama_sampler * smpl, llama_token token);
    LLAMA_API void                   llama_sampler_apply (      struct llama_sampler * smpl, llama_token_data_array * cur_p);
    LLAMA_API void                   llama_sampler_reset (      struct llama_sampler * smpl);
    LLAMA_API struct llama_sampler * llama_sampler_clone (const struct llama_sampler * smpl);
    // important: do not free if the sampler has been added to a llama_sampler_chain (via llama_sampler_chain_add)
    LLAMA_API void                   llama_sampler_free  (      struct llama_sampler * smpl);

    // llama_sampler_chain
    // a type of llama_sampler that can chain multiple samplers one after another

    LLAMA_API struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);

    // important: takes ownership of the sampler object and will free it when llama_sampler_free is called
    LLAMA_API void                   llama_sampler_chain_add(      struct llama_sampler * chain, struct llama_sampler * smpl);
    LLAMA_API struct llama_sampler * llama_sampler_chain_get(const struct llama_sampler * chain, int32_t i);
    LLAMA_API int                    llama_sampler_chain_n  (const struct llama_sampler * chain);

    // after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    LLAMA_API struct llama_sampler * llama_sampler_chain_remove(   struct llama_sampler * chain, int32_t i);

    // available samplers:

    LLAMA_API struct llama_sampler * llama_sampler_init_greedy(void);
    LLAMA_API struct llama_sampler * llama_sampler_init_dist  (uint32_t seed);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    /// Setting k <= 0 makes this a noop
    LLAMA_API struct llama_sampler * llama_sampler_init_top_k      (int32_t k);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API struct llama_sampler * llama_sampler_init_top_p      (float   p, size_t min_keep);

    /// @details Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
    LLAMA_API struct llama_sampler * llama_sampler_init_min_p      (float   p, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    LLAMA_API struct llama_sampler * llama_sampler_init_typical    (float   p, size_t min_keep);

    /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    LLAMA_API struct llama_sampler * llama_sampler_init_temp       (float   t);

    /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
    LLAMA_API struct llama_sampler * llama_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    LLAMA_API struct llama_sampler * llama_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);

    /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    LLAMA_API struct llama_sampler * llama_sampler_init_top_n_sigma(float   n);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat(
                             int32_t   n_vocab,
                            uint32_t   seed,
                               float   tau,
                               float   eta,
                             int32_t   m);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat_v2(
                            uint32_t   seed,
                               float   tau,
                               float   eta);

    /// @details Intializes a GBNF grammar, see grammars/README.md for details.
    /// @param vocab The vocabulary that this grammar will be used with.
    /// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
    /// @param grammar_root The name of the start symbol for the grammar.
    LLAMA_API struct llama_sampler * llama_sampler_init_grammar(
            const struct llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root);

/// @details Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
    /// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
    /// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
    LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy_patterns(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens);


    /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
    LLAMA_API struct llama_sampler * llama_sampler_init_penalties(
                             int32_t   penalty_last_n,   // last n tokens to penalize (0 = disable penalty, -1 = context size)
                               float   penalty_repeat,   // 1.0 = disabled
                               float   penalty_freq,     // 0.0 = disabled
                               float   penalty_present); // 0.0 = disabled

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    LLAMA_API struct llama_sampler * llama_sampler_init_dry(
            const struct llama_vocab *  vocab,
                             int32_t    n_ctx_train,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n,
                          const char ** seq_breakers,
                              size_t    num_breakers);

    LLAMA_API struct llama_sampler * llama_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const llama_logit_bias * logit_bias);

    // this sampler is meant to be used for fill-in-the-middle infilling
    // it's supposed to be used after top_k + top_p sampling
    //
    // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    // 2. combine probs of tokens that have the same prefix
    //
    // example:
    //
    // - before:
    //   "hel":   0.5
    //   "hell":  0.2
    //   "hello": 0.1
    //   "dummy": 0.1
    //
    // - after:
    //   "hel":   0.8
    //   "dummy": 0.1
    //
    // 3. discard non-EOG tokens with low prob
    // 4. if no tokens are left -> pick EOT
    //
    LLAMA_API struct llama_sampler * llama_sampler_init_infill(const struct llama_vocab * vocab);

    // Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise
    LLAMA_API uint32_t llama_sampler_get_seed(const struct llama_sampler * smpl);

    /// @details Sample and accept a token from the idx-th output of the last evaluation
    //
    // Shorthand for:
    //    const auto * logits = llama_get_logits_ith(ctx, idx);
    //    llama_token_data_array cur_p = { ... init from logits ... };
    //    llama_sampler_apply(smpl, &cur_p);
    //    auto token = cur_p.data[cur_p.selected].id;
    //    llama_sampler_accept(smpl, token);
    //    return token;
    // Returns the sampled token
    LLAMA_API llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);

    // TODO: extend in the future
    //LLAMA_API void llama_decode_with_sampler(struct llama_context * ctx, struct llama_sampler * smpl, struct llama_batch batch, ...);

    //
    // Model split
    //

    /// @details Build a split GGUF final path for this chunk.
    ///          llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    //  Returns the split_path length.
    LLAMA_API int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count);

    /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    ///          llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    //  Returns the split_prefix length.
    LLAMA_API int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count);

    // Print system information
    LLAMA_API const char * llama_print_system_info(void);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    LLAMA_API void llama_log_set(ggml_log_callback log_callback, void * user_data);

*/
