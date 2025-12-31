package core

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/kylerohn/gollama/internal/adapters"
)
import "C"

type Engine struct {
	model   Model
	vocab   Vocab
	context Context
	sampler Sampler
}

// Initialize model and system controlled backend
// TODO add better parameter manipulation
func InitializeModelDynamicBackend(filepath string, numGPULayers int32, numCtx uint32) Engine {
	adapters.GGMLBackendLoadAll()

	modelParams := adapters.ModelDefaultParams()
	adapters.SetNumGPULayers(&modelParams, numGPULayers)

	model := LoadModelFromFile(filepath, modelParams)
	vocab := model.GetVocab()

	ctxParams := adapters.ContextDefaultParams()
	adapters.SetNumCtx(&ctxParams, numCtx)
	adapters.SetNumBatch(&ctxParams, numCtx)

	context := NewContextFromModel(model, ctxParams)

	smplParams := adapters.SamplerChainDefaultParams()
	sampler := NewSamplerChain(smplParams)
	sampler.AddSamplerMinP(0.05, 1)
	sampler.AddSamplerTemp(0.8)
	sampler.AddSamplerDist(42)

	return Engine{model, vocab, context, sampler}
}

func (engine Engine) GenerateSync(prompt string, print bool) string {
	isFirst := engine.context.IsFirstTurn()
	tokens := engine.vocab.Tokenize(prompt, isFirst, true)

	batch := NewSingleBatch(tokens)
	defer adapters.FreeBatch(batch)

	response := ""

	for true {
		if engine.context.IsContextFull(len(tokens)) {
			fmt.Printf("Context Size Exceeded, exiting...\n")
			os.Exit(1)
		}

		engine.context.DecodeBatch(batch)
		newToken := engine.sampler.Sample(&engine.context, -1)

		if engine.vocab.IsEOG(newToken) {
			break
		}

		piece := engine.vocab.TokenToPiece(newToken, 0, true)
		if print {
			fmt.Print(piece)
		}
		response += piece

		newTokens := []Token{newToken}
		batch = NewSingleBatch(newTokens)
	}

	if print {
		fmt.Print("\n")
	}

	return response
}

func (engine Engine) SimpleCliChat() {
	messages := make([]adapters.ChatMessage, 0, 128)

	reader := bufio.NewReader(os.Stdin)

	for true {
		fmt.Printf("\033[32m> \033[0m")
		text, _ := reader.ReadString('\n')
		text = strings.ReplaceAll(text, "\n", "")
		if len(text) < 1 {
			break
		}

		tmpl := engine.model.ChatTemplate("")
		messages = append(messages, adapters.ChatMessage{Role: "user", Content: text})

		prompt := adapters.ApplyChatTemplate(tmpl, messages, true)
		resp := engine.GenerateSync(prompt, true)

		messages = append(messages, adapters.ChatMessage{Role: "assistant", Content: resp})
	}

	engine.sampler.Free()
	engine.context.Free()
	engine.model.Free()

}
