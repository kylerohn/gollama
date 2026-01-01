package main

import core "github.com/kylerohn/gollama/gollama/core"

func main() {
	path := "/home/kyle/projects/llm/models/gguf/Qwen3-4B-Q8_0.gguf"

	eng := core.InitializeModelDynamicBackend(path, 99, 32768)
	eng.SimpleCliChat()
}
