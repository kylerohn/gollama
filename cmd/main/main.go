package main

import core "github.com/kylerohn/gollama/gollama/core"

func main() {
	path := "/home/kyle/projects/llm/models/gguf/DeepSeek-R1-Distill-Qwen-1.5B-BF16.gguf"

	eng := core.InitializeModelDynamicBackend(path, 99, 1028)
	eng.SimpleCliChat()
}
