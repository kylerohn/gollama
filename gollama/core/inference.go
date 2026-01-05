package core

import (
	"bufio"
	"fmt"
	"log"
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
func InitializeModelDynamicBackend(filepath string, modelConfig ModelConfig, contextConfig ContextConfig, sampler Sampler) Engine {
	adapters.GGMLBackendLoadAll()

	modelParams := adapters.ModelDefaultParams()
	SetModelParams(&modelConfig, &modelParams)

	model := LoadModelFromFile(filepath, modelParams)
	vocab := model.GetVocab()

	ctxParams := adapters.ContextDefaultParams()
	SetContextParams(&contextConfig, &ctxParams)
	context := NewContextFromModel(model, ctxParams)

	return Engine{model, vocab, context, sampler}
}

func (engine Engine) GenerateSync(prompt string, print bool) string {
	isFirst := engine.context.IsFirstTurn()
	tokens, err := engine.vocab.Tokenize(prompt, isFirst, true)
	if err != nil {
		log.Fatal(err.Error())
	}

	batch := adapters.InitBatch(int32(len(tokens)), 0, 1)
	defer adapters.FreeBatch(batch)

	adapters.LoadSingleBatch(&batch, &tokens)

	response := ""

	for true {
		if engine.context.IsContextFull(len(tokens)) {
			fmt.Printf("Context Size Exceeded, exiting...\n")
			os.Exit(1)
		}

		engine.context.DecodeBatch(batch)
		nextToken := engine.sampler.Sample(&engine.context, -1)

		if engine.vocab.IsEOG(nextToken) {
			break
		}

		piece, err := engine.vocab.TokenToPiece(nextToken, 0, true)

		if err != nil {
			log.Fatal(err.Error())
		}

		if print {
			fmt.Print(piece)
		}
		response += piece

		nextTokens := []Token{nextToken}
		adapters.LoadSingleBatch(&batch, &nextTokens)
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

		tmpl, err := engine.model.ChatTemplate("")
		if err != nil {
			log.Fatal(err.Error())
		}
		messages = append(messages, adapters.ChatMessage{Role: "user", Content: text})

		prompt := adapters.ApplyChatTemplate(tmpl, messages, true)
		resp := engine.GenerateSync(prompt, true)

		messages = append(messages, adapters.ChatMessage{Role: "assistant", Content: resp})
	}

	engine.sampler.Free()
	engine.context.Free()
	engine.model.Free()

}
