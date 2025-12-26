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
	"runtime"
	"unsafe"
)

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


func SamplerName(smpl *Sampler) string {
	cStr := C.llama_sampler_name(smpl)

	return C.GoString(cStr)
}

func SamplerAccept(smpl *Sampler, token TokenT) {
	C.llama_sampler_accept(smpl, token)
}

func SamplerApply(smpl *Sampler, cur_p *TokenDataArray) {
	C.llama_sampler_apply(smpl, cur_p)
}

func SamplerReset(smpl *Sampler) {
	C.llama_sampler_reset(smpl)
}

func SamplerClone(smpl *Sampler) *Sampler {
	return C.llama_sampler_clone(smpl)
}

// important: do not free if the sampler has been added to a llama_sampler_chain (via llama_sampler_chain_add)
func SamplerFree(smpl *Sampler) {
	C.llama_sampler_free(smpl)
}


// llama_sampler_chain
// a type of llama_sampler that can chain multiple samplers one after another
// important: takes ownership of the sampler object and will free it when llama_sampler_free is called

func SamplerChainAdd(chain *Sampler, smpl *Sampler) {
	C.llama_sampler_chain_add(chain, smpl)
}

func SamplerChainGet(chain *Sampler, i int32) *Sampler {
	return C.llama_sampler_chain_get(chain, C.int32_t(i))
}

func SamplerChainN(chain *Sampler) int {
	return int(C.llama_sampler_chain_n(chain))
}

// after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
func SamplerChainRemove(chain *Sampler, i int32) *Sampler {
	return C.llama_sampler_chain_remove(chain, C.int32_t(i))
}

// available samplers:

func SamplerInitGreedy() *Sampler {
	return C.llama_sampler_init_greedy()
}

func SamplerInitDist(seed uint32) *Sampler {
	return C.llama_sampler_init_dist(C.uint32_t(seed))
}

// Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
// Setting k <= 0 makes this a noop
func SamplerInitTopK(seed int32) *Sampler {
	return C.llama_sampler_init_top_k(C.int32_t(seed))
}

// Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
func SamplerInitTopP(p float32, minKeep uint) *Sampler {
	return C.llama_sampler_init_top_p(C.float(p), C.size_t(minKeep))
}

// Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
func SamplerInitMinP(p float32, minKeep uint) *Sampler {
	return C.llama_sampler_init_min_p(C.float(p), C.size_t(minKeep))
}

// Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
func SamplerInitTypical(p float32, minKeep uint) *Sampler {
	return C.llama_sampler_init_typical(C.float(p), C.size_t(minKeep))
}

// Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
func SamplerInitTemp(t float32) *Sampler {
	return C.llama_sampler_init_temp(C.float(t))
}

// Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
func SamplerInitTempExt(t float32, delta float32, exponent float32) *Sampler {
	return C.llama_sampler_init_temp_ext(C.float(t), C.float(delta), C.float(exponent))
}

// XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
func SamplerInitXTC(p float32, t float32, minKeep uint, seed uint32) *Sampler {
	return C.llama_sampler_init_xtc(C.float(p), C.float(t), C.size_t(minKeep), C.uint32_t(seed))
}

// Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
func SamplerInitTopNSigma(n float32) *Sampler {
	return C.llama_sampler_init_top_n_sigma(C.float(n))
}

// Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
//
//	candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
//
// tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
// eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
// m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
// mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
func SamplerInitMirostat(n_vocab int32, seed uint32, tau float32, eta float32, m int32) *Sampler {
	return C.llama_sampler_init_mirostat(C.int32_t(n_vocab), C.uint32_t(seed), C.float(tau), C.float(eta), C.int32_t(m))
}

// Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
// candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
// tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
// eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
// mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
func SamplerInitMirostatV2(seed uint32, tau float32, eta float32) *Sampler {
	return C.llama_sampler_init_mirostat_v2(C.uint32_t(seed), C.float(tau), C.float(eta))
}

// Intializes a GBNF grammar, see grammars/README.md for details.
// vocab The vocabulary that this grammar will be used with.
// grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
// grammar_root The name of the start symbol for the grammar.
func SamplerInitGrammar(vocab *Vocab, grammarStr string, grammarRoot string) *Sampler {
	cStrGrammarStr := C.CString(grammarStr)
	defer C.free(unsafe.Pointer(cStrGrammarStr))

	cStrGrammarRoot := C.CString(grammarRoot)
	defer C.free(unsafe.Pointer(cStrGrammarRoot))

	return C.llama_sampler_init_grammar(vocab, cStrGrammarStr, cStrGrammarRoot)
}

// TODO later, i'm feeling lazy
// Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
// trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
// trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token
func SamplerInitGrammarLazyPatterns(vocab *Vocab, grammarStr string, grammarRoot string, triggerPatterns []string, triggerTokens []TokenT) *Sampler {
	cStrTriggerPatterns := make([]*C.char, 0, len(triggerPatterns))

	for _, val := range triggerPatterns {
		if val != "" {
			cPattern := C.CString(val)
			defer C.free(unsafe.Pointer(cPattern))
			cStrTriggerPatterns = append(cStrTriggerPatterns, cPattern)
		}
	}

	cStrGrammarStr := C.CString(grammarStr)
	defer C.free(unsafe.Pointer(cStrGrammarStr))

	cStrGrammarRoot := C.CString(grammarRoot)
	defer C.free(unsafe.Pointer(cStrGrammarRoot))

	var patternsPtr **C.char
	if len(cStrTriggerPatterns) > 0 {
		patternsPtr = (**C.char)(unsafe.Pointer(&cStrTriggerPatterns[0]))
	}

	var tokensPtr *TokenT
	if len(triggerTokens) > 0 {
		tokensPtr = (*TokenT)(unsafe.Pointer(&triggerTokens[0]))
	}

	out := C.llama_sampler_init_grammar_lazy_patterns(
		vocab,
		cStrGrammarStr,
		cStrGrammarRoot,
		patternsPtr,
		C.size_t(len(cStrTriggerPatterns)),
		(*C.llama_token)(unsafe.Pointer(tokensPtr)), // or adjust if TokenT aliases C.llama_token
		C.size_t(len(triggerTokens)),
	)

	runtime.KeepAlive(cStrTriggerPatterns)

	return out
}

// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
// penaltyLastN: last n tokens to penalize (0 = disable penalty, -1 = context size)
// penaltyRepeat: 1.0 = disabled
// penaltyFreq: 0.0 = disabled
// penaltyPresent: 0.0 = disabled
func SamplerInitPenalties(penaltyLastN int32, penaltyRepeat float32, penaltyFreq float32, penaltyPresent float32) *Sampler {
	return C.llama_sampler_init_penalties(C.int32_t(penaltyLastN), C.float(penaltyRepeat), C.float(penaltyFreq), C.float(penaltyPresent))
}

// DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
func SamplerInitDry(vocab *Vocab, nCtxTrain int32, dryMultiplier float32, dryBase float32, dryAllowedLength int32, dryPenaltyLastN int32, seqBreakers []string) *Sampler {
	cStrSeqBreakers := make([]*C.char, 0, len(seqBreakers))

	for _, val := range seqBreakers {
		if val != "" {
			cPattern := C.CString(val)
			defer C.free(unsafe.Pointer(cPattern))
			cStrSeqBreakers = append(cStrSeqBreakers, cPattern)
		}
	}

	var breakersPtr **C.char
	if len(cStrSeqBreakers) > 0 {
		breakersPtr = (**C.char)(unsafe.Pointer(&cStrSeqBreakers[0]))
	}

	out := C.llama_sampler_init_dry(
		vocab,
		C.int32_t(nCtxTrain),
		C.float(dryMultiplier),
		C.float(dryBase),
		C.int32_t(dryAllowedLength),
		C.int32_t(dryPenaltyLastN),
		breakersPtr,
		C.size_t(len(cStrSeqBreakers)),
	)

	runtime.KeepAlive(cStrSeqBreakers)

	return out
}

func SamplerInitLogitBias(nVocab int32, nLogitBias int32, logitBias *LogitBias) *Sampler {
	return C.llama_sampler_init_logit_bias(
		C.int32_t(nVocab),
		C.int32_t(nLogitBias),
		logitBias,
	)
}

// this sampler is meant to be used for fill-in-the-middle infilling
// it's supposed to be used after top_k + top_p sampling
//
// 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
// 2. combine probs of tokens that have the same prefix
//
// example:
//
//   - before:
//     "hel":   0.5
//     "hell":  0.2
//     "hello": 0.1
//     "dummy": 0.1
//
//   - after:
//     "hel":   0.8
//     "dummy": 0.1
//
// 3. discard non-EOG tokens with low prob
// 4. if no tokens are left -> pick EOT
func SamplerInitInfill(vocab *Vocab) *Sampler {
	return C.llama_sampler_init_infill(vocab)
}

// Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise
func SamplerGetSeed(smpl *Sampler) uint32 {
	return uint32(C.llama_sampler_get_seed(smpl))
}


// Sample and accept a token from the idx-th output of the last evaluation
//
// Shorthand for:
//    const auto * logits = llama_get_logits_ith(ctx, idx);
//    llama_token_data_array cur_p = { ... init from logits ... };
//    llama_sampler_apply(smpl, &cur_p);
//    auto token = cur_p.data[cur_p.selected].id;
//    llama_sampler_accept(smpl, token);
//    return token;
// Returns the sampled token
func SamplerSample(smpl *Sampler, ctx *Context, idx int32) TokenT{
	return C.llama_sampler_sample(smpl, ctx, C.int32_t(idx))
}
// TODO: extend in the future (look for changes in llama.cpp)
//LLAMA_API void llama_decode_with_sampler(struct llama_context * ctx, struct llama_sampler * smpl, struct llama_batch batch, ...);
