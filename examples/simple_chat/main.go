package simple_chat

import core "github.com/kylerohn/gollama/gollama/core"

func main() {
	path := "/home/krohn/projects/models/deepseek-qwen/DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf"

	eng := core.InitializeModelDynamicBackend(path, 99, 32768)
	eng.SimpleCliChat()
}
