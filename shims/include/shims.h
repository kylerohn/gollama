#pragma once
#include <llama.h>

typedef struct llama_vocab llama_vocab;
typedef struct llama_batch llama_batch;
typedef struct llama_chat_message llama_chat_message;

#ifdef __cplusplus
extern "C"
{
#endif

    // create llama_token vector
    llama_batch shim_tokenization_batching(llama_vocab *vocab, const char *prompt, int prompt_size, int n_prompt_tokens, bool is_first);
    llama_chat_message shim_create_chat_message(const char *role, const char *message);
    

#ifdef __cplusplus
} // extern "C"
#endif