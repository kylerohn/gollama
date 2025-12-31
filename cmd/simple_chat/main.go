package main

/*
#cgo CFLAGS: -I${SRCDIR}/../external/llama.cpp/include
#cgo CFLAGS: -I${SRCDIR}/../external/llama.cpp/ggml/include
#cgo CFLAGS: -I${SRCDIR}/../shims/include
#cgo LDFLAGS: -L${SRCDIR}/../build/bin -lggml -lllama -lshims -Wl,-rpath,$ORIGIN/build/bin

#include <stdio.h>
#include <ggml.h>
#include <llama.h>
#include <shims.h>
*/
import "C"
import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"unsafe"
)

func main() {
	const n_gpu_layers = 99
	const n_ctx = 2048

	C.ggml_backend_load_all()

	var model_params C.struct_llama_model_params = C.llama_model_default_params()
	model_params.n_gpu_layers = n_gpu_layers

	var model *C.struct_llama_model = C.llama_model_load_from_file(C.CString("/home/krohn/projects/models/deepseek-qwen/DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf"), model_params)
	if model == nil {
		log.Fatal("Failed to load model")
		os.Exit(1)
	}

	var vocab *C.struct_llama_vocab = C.llama_model_get_vocab(model)

	// initialize the context
	var ctx_params C.struct_llama_context_params = C.llama_context_default_params()
	ctx_params.n_ctx = n_ctx
	ctx_params.n_batch = n_ctx

	var ctx *C.struct_llama_context = C.llama_init_from_model(model, ctx_params)

	// initialize the sampler
	var smpl *C.struct_llama_sampler = C.llama_sampler_chain_init(C.llama_sampler_chain_default_params())
	C.llama_sampler_chain_add(smpl, C.llama_sampler_init_min_p(0.05, 1))
	C.llama_sampler_chain_add(smpl, C.llama_sampler_init_temp(0.8))
	C.llama_sampler_chain_add(smpl, C.llama_sampler_init_dist(C.LLAMA_DEFAULT_SEED))

	generate := func(prompt string) string {
		response := ""
		is_first := C.llama_memory_seq_pos_max(C.llama_get_memory(ctx), 0) == -1

		// tokenize
		n_prompt_tokens := -C.llama_tokenize(vocab, C.CString(prompt), C.int(len(prompt)), nil, 0, C.bool(is_first), true)

		prompt_tokens := make([]C.llama_token, n_prompt_tokens)

		if int(C.llama_tokenize(vocab, C.CString(prompt), C.int(len(prompt)), (*C.llama_token)(unsafe.Pointer(&prompt_tokens[0])), n_prompt_tokens, C.bool(is_first), true)) < 0 {
			log.Fatal("Failed to tokenize")
		}

		var batch C.llama_batch = C.llama_batch_get_one((*C.llama_token)(unsafe.Pointer(&prompt_tokens[0])), n_prompt_tokens)

		for true {
			n_ctx := C.llama_n_ctx(ctx)
			n_ctx_used := C.llama_memory_seq_pos_max(C.llama_get_memory(ctx), 0) + 1
			if int(n_ctx_used)+int(batch.n_tokens) > int(n_ctx) {
				log.Fatal("\033[0m\ncontext size exceeded\n")
			}
			ret := C.llama_decode(ctx, batch)
			if int(ret) != 0 {
				log.Fatal("\033[0m\nfailed to decode\n")
			}

			// sample new token
			new_token_id := C.llama_sampler_sample(smpl, ctx, -1)

			// check for end of generation
			if bool(C.llama_vocab_is_eog(vocab, new_token_id)) {
				break
			}

			buf := make([]byte, 256)
			n := C.llama_token_to_piece(vocab, new_token_id, (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf)), 0, true)
			if int(n) < 0 {
				log.Fatal("\033[0m\nfailed to convert token to piece\n")
			}

			piece := string(buf)
			fmt.Printf(piece)
			response += piece

			batch = C.llama_batch_get_one(&new_token_id, 1)
		}

		return response

	}

	var messages []C.struct_llama_chat_message
	var formatted = make([]uint8, int(C.llama_n_ctx(ctx)))
	prev_len := 0

	reader := bufio.NewReader(os.Stdin)

	for true {
		fmt.Printf("\033[32m> \033[0m")
		text, _ := reader.ReadString('\n')
		text = strings.ReplaceAll(text, "\n", "")

		if len(text) < 1 {
			break
		}

		tmpl := C.llama_model_chat_template(model, nil)

		messages = append(messages, C.shim_create_chat_message(C.CString("user"), C.CString(text)))
		new_len := C.llama_chat_apply_template(tmpl, (*C.struct_llama_chat_message)(unsafe.Pointer(&messages[0])), C.size_t(len(messages)), true, (*C.char)(unsafe.Pointer(&formatted[0])), C.int(len(formatted)))
		if int(new_len) > len(formatted) {
			s := make([]uint8, new_len)
			copy(s, formatted)
			formatted = s
		}

		if int(new_len) < 0 {
			log.Fatal("\033[0m\nfailed to apply chat template\n")
		}

		prompt := string(formatted[prev_len:new_len])

		fmt.Printf("\033[33m")
		response := generate(prompt)
		fmt.Printf("\n\033[0m")

		messages = append(messages, C.shim_create_chat_message(C.CString("assistant"), C.CString(response)))
		prev_len = int(C.llama_chat_apply_template(tmpl, (*C.struct_llama_chat_message)(unsafe.Pointer(&messages[0])), C.size_t(len(messages)), false, nil, 0))
		if int(prev_len) < 0 {
			log.Fatal("\033[0m\nfailed to apply chat template\n")
		}
	}

	C.llama_sampler_free(smpl)
	C.llama_free(ctx)
	C.llama_model_free(model)

}
