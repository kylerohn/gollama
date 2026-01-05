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
	"runtime"
	"unsafe"
)

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

func FreeModel(model *Model) {
	C.llama_model_free(model)
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

// Metadata value as a string by key name
func ModelMetaValStr(model *Model, key string) (string, error) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	// TODO Add Bounded Growth
	var size uint = 16384
	buf := make([]byte, size)

	strLen := int32(C.llama_model_meta_val_str(model, cKey, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(size)))

	if strLen < 0 {
		return "", fmt.Errorf("ModelMetaValStr: failed to retrieve key %s", key)
	}
	if uint(strLen) < size-1 {
		res := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
		runtime.KeepAlive(buf)
		return res, nil
	}

	return "", fmt.Errorf("ModelMetaValStr: result length too large for buffer (%d > %d)", strLen, size)
}

// get number of metadata key/value pairs
func ModelMetaCount(model *Model) int32 {
	return int32(C.llama_model_meta_count(model))
}

// get metadata key name by index
func ModelMetaKeyByIndex(model *Model, i int) (string, error) {
	// TODO Add Bounded Growth
	var size uint = 16384
	buf := make([]byte, size)

	strLen := int32(C.llama_model_meta_key_by_index(model, C.int(i), (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(size)))

	if strLen < 0 {
		return "", fmt.Errorf("ModelMetaKeyByIndex: failed to retrieve key at index %d", i)
	}
	if uint(strLen) < size-1 {
		res := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
		runtime.KeepAlive(buf)
		return res, nil
	}

	return "", fmt.Errorf("ModelMetaKeyByIndex: result length too large for buffer (%d > %d)", strLen, size)
}

// get metadata key value by index
func ModelMetaValByIndex(model *Model, i int) (string, error) {
	// TODO Add Bounded Growth
	var size uint = 16384
	buf := make([]byte, size)

	strLen := int32(C.llama_model_meta_val_str_by_index(model, C.int(i), (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(size)))

	if strLen < 0 {
		return "", fmt.Errorf("ModelMetaValByIndex: failed to retrieve value at index %d", i)
	}
	if uint(strLen) < size-1 {
		res := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
		runtime.KeepAlive(buf)
		return res, nil
	}

	return "", fmt.Errorf("ModelMetaValByIndex: result length too large for buffer (%d > %d)", strLen, size)
}

// Get a string describing the model type
func ModelDesc(model *Model) (string, error) {
	// TODO Add Bounded Growth
	var size uint = 16384
	buf := make([]byte, size)

	strLen := int32(C.llama_model_desc(model, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(size)))

	if strLen < 0 {
		return "", fmt.Errorf("ModelDesc: failed to retrieve model description metadata")
	}
	if uint(strLen) < size-1 {
		res := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
		runtime.KeepAlive(buf)
		return res, nil
	}

	return "", fmt.Errorf("ModelDesc: result length too large for buffer (%d > %d)", strLen, size)
}

// Returns the total size of all the tensors in the model in bytes
func ModelSize(model *Model) uint64 {
	return uint64(C.llama_model_size(model))
}

// Get the default chat template. Returns nullptr if not available
// If name is "", returns the default chat template
func ModelChatTemplate(model *Model, name string) (string, error) {

	var res *C.char

	if name != "" {
		cStr := C.CString(name)
		defer C.free(unsafe.Pointer(cStr))
		res = C.llama_model_chat_template(model, cStr)

	} else {
		res = C.llama_model_chat_template(model, nil)
	}

	if res == nil {
		if name == "" {
			name = "[DEFAULT]"
		}
		return "", fmt.Errorf("ModelChatTemplate: could not find chat template for name %s", name)
	}

	return C.GoString(res), nil
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

// Load a LoRA adapter from file
func InitAdapterLoRA(model *Model, loraPath string) (*AdapterLoRA, error) {
	cPath := C.CString(loraPath)
	defer C.free(unsafe.Pointer(cPath))

	adapter := C.llama_adapter_lora_init(model, cPath)
	if adapter == nil {
		return nil, fmt.Errorf("InitAdapterLoRA: failed Loading LoRA adapter from path %s", loraPath)
	}
	return adapter, nil
}
