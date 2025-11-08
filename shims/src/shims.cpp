#include "shims.h"
#include <vector>
#include <string>
#include <llama.h>

extern "C" {
    llama_batch shim_tokenization_batching(llama_vocab *vocab, const char *prompt, int prompt_size, int n_prompt_tokens, bool is_first){
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(vocab, prompt, prompt_size, prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
            GGML_ABORT("failed to tokenize the prompt\n");
        }

        // prepare a batch for the prompt
        return llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    }
    
    llama_chat_message shim_create_chat_message(const char *role, const char *message){
        return llama_chat_message{
            role,
            message
        };
    }
}