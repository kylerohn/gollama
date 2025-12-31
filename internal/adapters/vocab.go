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
func Tokenize(vocab *Vocab, text string, addSpecial bool, parseSpecial bool) []TokenT {
	cStr := C.CString(text)
	defer C.free(unsafe.Pointer(cStr))

	nTokensMax := -C.llama_tokenize(vocab, cStr, C.int32_t(len(text)), nil, 0, C.bool(addSpecial), C.bool(parseSpecial))

	buf := make([]TokenT, nTokensMax)
	ok := C.llama_tokenize(vocab, cStr, C.int32_t(len(text)), (*TokenT)(unsafe.Pointer(&buf[0])), nTokensMax, C.bool(addSpecial), C.bool(parseSpecial))

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
