package main

import core "github.com/kylerohn/gollama/gollama/core"

func main() {
	path := "/home/krohn/projects/models/deepseek-qwen/DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf"

	var ctxSize uint32 = 32768

	modelConfig := core.DefaultModelConfig()

	contextConfig := core.DefaultContextConfig()
	contextConfig.NumCtx = ctxSize
	contextConfig.NumBatch = ctxSize

	sampler := core.NewSamplerChain(true)
	sampler.AddSamplerMinP(0.05, 1)
	sampler.AddSamplerTemp(0.8)
	sampler.AddSamplerDist(42)

	eng := core.InitializeModelDynamicBackend(path, modelConfig, contextConfig, sampler)
	eng.SimpleCliChat()
}
