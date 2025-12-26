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
	"log"
	"runtime"
	"unsafe"
)

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

// Get list of built-in chat templates
// LLAMA_API int32_t llama_chat_builtin_templates(const char ** output, size_t len);
func GetChatTemplates() []string {

	n := C.llama_chat_builtin_templates(nil, 0)

	if n <= 0 {
		return nil
	}

	buf := make([]*C.char, int(n))

	written := C.llama_chat_builtin_templates((**C.char)(unsafe.Pointer(&buf[0])), C.size_t(n))

	if written <= 0 {
		return nil
	}

	count := max(int(written), len(buf))

	out := make([]string, 0, count)
	for i := range count {
		if buf[i] == nil {
			continue
		}

		out = append(out, C.GoString(buf[i]))
	}

	runtime.KeepAlive(buf)

	return out
}

//
// Sampling
//

// mirror of llama_sampler_i:
func SamplerInit(iface *SamplerI, ctx SamplerContextT) *Sampler {
	return C.llama_sampler_init(iface, ctx)
}

func SamplerChainInit(params SamplerChainParams) *Sampler {
	return C.llama_sampler_chain_init(params)
}

// Print system information
func GetSystemInformation() string {
	return C.GoString(C.llama_print_system_info())
}

//
// Model split
//

// TODO bridge this
// Build a split GGUF final path for this chunk.
// llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
// Returns the split_path length.
// LLAMA_API int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count);

// TODO and this
// Extract the path prefix from the split_path if and only if the split_no and split_count match.
// llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
// Returns the split_prefix length.
// LLAMA_API int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count);

// TODO idk how to do this
// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
//    LLAMA_API void llama_log_set(ggml_log_callback log_callback, void * user_data);


//
// Performance utils
//
// NOTE: Used by llama.cpp examples/tools, avoid using in third-party apps. Instead, do your own performance measurements.
//

/*
    LLAMA_API struct llama_perf_context_data llama_perf_context      (const struct llama_context * ctx);
    LLAMA_API void                           llama_perf_context_print(const struct llama_context * ctx);
    LLAMA_API void                           llama_perf_context_reset(      struct llama_context * ctx);

    // NOTE: the following work only with samplers constructed via llama_sampler_chain_init
    LLAMA_API struct llama_perf_sampler_data llama_perf_sampler      (const struct llama_sampler * chain);
    LLAMA_API void                           llama_perf_sampler_print(const struct llama_sampler * chain);
    LLAMA_API void                           llama_perf_sampler_reset(      struct llama_sampler * chain);

    // print a breakdown of per-device memory use via LLAMA_LOG:
    LLAMA_API void llama_memory_breakdown_print(const struct llama_context * ctx);

    //
    // training
    //

    // function that returns whether or not a given tensor contains trainable parameters
    typedef bool (*llama_opt_param_filter)(const struct ggml_tensor * tensor, void * userdata);

    // always returns true
    LLAMA_API bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata);

    struct llama_opt_params {
        uint32_t n_ctx_train; // assumed context size post training, use context size specified in llama_context if 0

        llama_opt_param_filter param_filter; // callback for determining which tensors contain trainable parameters
        void * param_filter_ud;              // userdata for determining which tensors contain trainable parameters

        ggml_opt_get_optimizer_params get_opt_pars; // callback for calculating optimizer parameters
        void * get_opt_pars_ud;                     // userdata for calculating optimizer parameters

        enum ggml_opt_optimizer_type optimizer_type;
    };

    LLAMA_API void llama_opt_init(struct llama_context * lctx, struct llama_model * model, struct llama_opt_params lopt_params);

    LLAMA_API void llama_opt_epoch(
            struct llama_context    * lctx,
            ggml_opt_dataset_t        dataset,
            ggml_opt_result_t         result_train,
            ggml_opt_result_t         result_eval,
            int64_t                   idata_split,
            ggml_opt_epoch_callback   callback_train,
            ggml_opt_epoch_callback   callback_eval);
*/