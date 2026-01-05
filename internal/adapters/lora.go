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

// Metadata value as a string by key name
func AdapterMetaValStr(adapter *AdapterLoRA, key string) (string, error) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	// TODO Add Bounded Growth
	var size uint = 16384
	buf := make([]byte, size)

	strLen := int32(C.llama_adapter_meta_val_str(adapter, cKey, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(size)))

	if strLen < 0 {
		return "", fmt.Errorf("AdapterMetaValStr: failed to retrieve key %s", key)
	}
	if uint(strLen) < size-1 {
		res := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
		runtime.KeepAlive(buf)
		return res, nil
	}

	return "", fmt.Errorf("AdapterMetaValStr: result length too large for buffer (%d > %d)", strLen, size)
}

// get number of metadata key/value pairs
func AdapterMetaCount(adapter *AdapterLoRA) int32 {
	return int32(C.llama_adapter_meta_count(adapter))
}

// get metadata key name by index
func AdapterMetaKeyByIndex(adapter *AdapterLoRA, i int) (string, error) {
	// TODO Add Bounded Growth
	var size uint32 = 16384
	buf := make([]byte, size)

	strLen := int32(C.llama_adapter_meta_key_by_index(adapter, C.int(i), (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(size)))

	if strLen < 0 {
		return "", fmt.Errorf("AdapterMetaKeyByIndex: failed to retrieve key at index %d", i)
	}
	if uint32(strLen) < size-1 {
		res := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
		runtime.KeepAlive(buf)
		return res, nil
	}

	return "", fmt.Errorf("AdapterMetaKeyByIndex: result length too large for buffer (%d > %d)", strLen, size)
}

// get metadata key value by index
func AdapterMetaValByIndex(adapter *AdapterLoRA, i int) (string, error) {
	// TODO Add Bounded Growth
	var size uint = 16384
	buf := make([]byte, size)

	strLen := int32(C.llama_adapter_meta_val_str_by_index(adapter, C.int(i), (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(size)))

	if strLen < 0 {
		return "", fmt.Errorf("AdapterMetaValByIndex: failed to retrieve value at index %d", i)
	}
	if uint(strLen) < size-1 {
		res := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
		runtime.KeepAlive(buf)
		return res, nil
	}

	return "", fmt.Errorf("AdapterMetaValByIndex: result length too large for buffer (%d > %d)", strLen, size)
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
func AdapterGetALoRAInvocationTokens(adapter *AdapterLoRA) ([]TokenT, error) {
	tokenPtr := C.llama_adapter_get_alora_invocation_tokens(adapter)
	if tokenPtr == nil {
		return nil, fmt.Errorf("AdapterGetLoRAInvocationTokens: internal c method returned nullptr")
	}
	count := AdapterGetALoRANumInvocationTokens(adapter)
	res := unsafe.Slice((*TokenT)(tokenPtr), count)
	return res, nil
}
