package core

import "github.com/kylerohn/gollama/internal/adapters"

type Batch = adapters.Batch
type Token = adapters.TokenT

//
// ModelParams
//

type ModelParams struct {
	ptr *adapters.ModelParams
}

func NewDefaultModelParams() ModelParams {
	ptr := adapters.ModelDefaultParams()
	return ModelParams{&ptr}
}

//
// Context
//

type Context struct {
	ptr *adapters.Context
}

func NewContextFromModel(model Model, params adapters.ContextParams) Context {
	ptr := adapters.InitFromModel(model.ptr, params)
	return Context{ptr}
}

func (c *Context) Free() {
	adapters.Free(c.ptr)
}

func (c *Context) NumCtx() uint32 {
	return adapters.NumCtx(c.ptr)
}

func (c *Context) NumBatch() uint32 {
	return adapters.NumBatch(c.ptr)
}

// NumUBatch returns the size of the ubatch (batch of sequences) in the context.
func (c *Context) NumUBatch() uint32 {
	return adapters.NumUBatch(c.ptr)
}

// NumSeqMax returns the maximum number of sequences that can be processed in parallel.
func (c *Context) NumSeqMax() uint32 {
	return adapters.NumSeqMax(c.ptr)
}

func (c *Context) GetModel() *adapters.Model {
	return adapters.GetModel(c.ptr)
}

func (c *Context) GetMemory() Memory {
	ptr := adapters.GetMemory(c.ptr)
	return Memory{ptr}
}

// PoolingType returns the pooling type used for embeddings in the context.
func (c *Context) PoolingType() adapters.PoolingT {
	return adapters.PoolingType(c.ptr)
}

func (c *Context) SetAdapterLoRA(adapter *adapters.AdapterLoRA, scale float32) int32 {
	return adapters.SetAdapterLoRA(c.ptr, adapter, scale)
}

func (c *Context) RemoveAdapterLoRA(adapter *adapters.AdapterLoRA) {
	adapters.RemoveAdapterLoRA(c.ptr, adapter)
}

func (c *Context) ClearAdapterLoRA() {
	adapters.ClearAdapterLoRA(c.ptr)
}

// ApplyAdapterCVec applies a control vector to the context. If data is empty, clears the currently loaded vector.
// nEmbed is the size of a single layer's control, and data should be an nEmbed x nLayers buffer starting from layer 1.
// ilStart and ilEnd specify the layer range the vector applies to (both inclusive).
func (c *Context) ApplyAdapterCVec(data []float32, nEmbed int32, ilStart int32, ilEnd int32) {
	adapters.ApplyAdapterCVec(c.ptr, data, nEmbed, ilStart, ilEnd)
}

func (c *Context) GetStateSize() uint {
	return adapters.GetStateSize(c.ptr)
}

func (c *Context) CopyStateData() ([]byte, error) {
	return adapters.CopyStateData(c.ptr)
}

func (c *Context) SetStateData(src []byte) {
	adapters.SetStateData(c.ptr, src)
}

func (c *Context) LoadStateFromFile(statePath string) ([]Token, error) {
	return adapters.LoadStateFromFile(c.ptr, statePath)
}

func (c *Context) SaveStateToFile(statePath string, tokens []Token) {
	adapters.SaveStateToFile(c.ptr, statePath, tokens)
}

func (c *Context) GetSeqStateSize(seq_id int32) uint {
	return adapters.GetSeqStateSize(c.ptr, seq_id)
}

func (c *Context) GetSeqStateData(seq_id int32) []uint8 {
	return adapters.GetSeqStateData(c.ptr, seq_id)
}

func (c *Context) SetSeqStateData(src []uint8, dest_seq_id int32) {
	adapters.SetSeqStateData(c.ptr, src, dest_seq_id)
}

func (c *Context) SaveSeqStateToFile(statePath string, seq_id int32, tokens []Token) {
	adapters.SaveSeqStateToFile(c.ptr, statePath, seq_id, tokens)
}

func (c *Context) LoadSeqStateFromFile(statePath string, dest_seq_id int32) []Token {
	return adapters.LoadSeqStateFromFile(c.ptr, statePath, dest_seq_id)
}

func (c *Context) GetSeqStateSizeExt(seq_id int32, flags uint32) uint {
	return adapters.GetSeqStateSizeExt(c.ptr, seq_id, flags)
}

func (c *Context) GetSeqStateDataExt(seq_id int32, flags uint32) []uint8 {
	return adapters.GetSeqStateDataExt(c.ptr, seq_id, flags)
}

func (c *Context) SetSeqStateDataExt(src []uint8, dest_seq_id int32, flags uint32) {
	adapters.SetSeqStateDataExt(c.ptr, src, dest_seq_id, flags)
}

// EncodeBatch processes a batch of tokens without using KV cache. For encode-decoder models, uses the encoder.
func (c *Context) EncodeBatch(batch Batch) {
	adapters.EncodeBatch(c.ptr, batch)
}

// DecodeBatch processes a batch of tokens with KV cache. For encode-decoder models, uses the decoder.
func (c *Context) DecodeBatch(batch Batch) {
	adapters.DecodeBatch(c.ptr, batch)
}

func (c *Context) SetNumDecodingThreads(n_threads int32, n_threads_batch int32) {
	adapters.SetNumDecodingThreads(c.ptr, n_threads, n_threads_batch)
}

// GetNumDecodingThreads returns the number of threads used for single-token generation.
func (c *Context) GetNumDecodingThreads() int32 {
	return adapters.GetNumDecodingThreads(c.ptr)
}

// GetNumDecodingThreadsBatch returns the number of threads used for batch/prompt processing (multiple tokens).
func (c *Context) GetNumDecodingThreadsBatch() int32 {
	return adapters.GetNumDecodingThreadsBatch(c.ptr)
}

// SetEmbeddings enables or disables whether the context outputs embeddings.
func (c *Context) SetEmbeddings(embeddings bool) {
	adapters.SetEmbeddings(c.ptr, embeddings)
}

// SetCausalAttn enables or disables causal attention. When enabled, the model only attends to past tokens.
func (c *Context) SetCausalAttn(casual_attn bool) {
	adapters.SetCausalAttn(c.ptr, casual_attn)
}

// SetWarmup enables or disables warmup mode. When enabled, all model tensors are activated during decode() to load and cache their weights.
func (c *Context) SetWarmup(warmup bool) {
	adapters.SetWarmup(c.ptr, warmup)
}

// Synchronize waits until all GPU/accelerator computations are finished.
func (c *Context) Synchronize() {
	adapters.Synchronize(c.ptr)
}

// GetIthTokenLogits returns the logits for the i-th token. Negative indices access logits in reverse order (-1 is the last logit).
func (c *Context) GetIthTokenLogits(vocab *adapters.Vocab, i int32) []float32 {
	return adapters.GetIthTokenLogits(c.ptr, vocab, i)
}

// GetIthEmbeddings returns the embeddings for the i-th token. Negative indices access embeddings in reverse order (-1 is the last embedding).
func (c *Context) GetIthEmbeddings(model *adapters.Model, i int32) []float32 {
	return adapters.GetIthEmbeddings(c.ptr, model, i)
}

// GetIthSeqEmbeddings returns the embeddings for a specific sequence.
func (c *Context) GetIthSeqEmbeddings(model *adapters.Model, seq_id int32) []float32 {
	return adapters.GetIthSeqEmbeddings(c.ptr, model, seq_id)
}

func (c *Context) IsFirstTurn() bool {
	mem := c.GetMemory()
	return mem.SequenceMax(0) == -1
}

func (c *Context) IsContextFull(nTokens int) bool {
	numCtx := c.NumCtx()
	mem := c.GetMemory()
	numCtxUsed := mem.SequenceMax(0) + 1
	return (numCtxUsed + int32(nTokens)) > int32(numCtx)
}

//
// Model
//

type Model struct {
	ptr *adapters.Model
}

func (m *Model) Free() {
	adapters.FreeModel(m.ptr)
}

func LoadModelFromFile(filepath string, params adapters.ModelParams) Model {
	ptr := adapters.LoadModelFromFile(filepath, params)
	return Model{ptr}
}

func (m *Model) GetVocab() Vocab {
	ptr := adapters.GetModelVocab(m.ptr)
	return Vocab{ptr}
}

func (m *Model) RopeType() adapters.RopeT {
	return adapters.ModelRopeType(m.ptr)
}

// NumCtxTrain returns the context size the model was trained on.
func (m *Model) NumCtxTrain() int32 {
	return adapters.ModelNumCtxTrain(m.ptr)
}

// NumEmbd returns the embedding dimension size.
func (m *Model) NumEmbd() int32 {
	return adapters.ModelNumEmbd(m.ptr)
}

func (m *Model) NumLayer() int32 {
	return adapters.ModelNumLayer(m.ptr)
}

func (m *Model) NumHead() int32 {
	return adapters.ModelNumHead(m.ptr)
}

// NumHeadKV returns the number of key-value heads (may be less than NumHead for grouped query attention).
func (m *Model) NumHeadKV() int32 {
	return adapters.ModelNumHeadKV(m.ptr)
}

// NumSwa returns the number of sliding window attention layers (if applicable).
func (m *Model) NumSwa() int32 {
	return adapters.ModelNumSwa(m.ptr)
}

// RopeFreqScaleTrain returns the model's RoPE (Rotary Position Embedding) frequency scaling factor.
func (m *Model) RopeFreqScaleTrain() float32 {
	return adapters.ModelRopeFreqScaleTrain(m.ptr)
}

// NumClsOut returns the number of classifier outputs (for classifier models only).
func (m *Model) NumClsOut() uint32 {
	return adapters.ModelNumClsOut(m.ptr)
}

// ClsLabel returns the label for a classifier output by index (classifier models only).
func (m *Model) ClsLabel(i uint32) string {
	return adapters.ModelClsLabel(m.ptr, i)
}

// MetaValStr retrieves a metadata value as a string by key name.
func (m *Model) MetaValStr(key string) (string, error) {
	return adapters.ModelMetaValStr(m.ptr, key)
}

// MetaCount returns the number of metadata key-value pairs.
func (m *Model) MetaCount() int32 {
	return adapters.ModelMetaCount(m.ptr)
}

// MetaKeyByIndex returns the metadata key name at the specified index.
func (m *Model) MetaKeyByIndex(i int) (string, error) {
	return adapters.ModelMetaKeyByIndex(m.ptr, i)
}

// MetaValByIndex returns the metadata value at the specified index.
func (m *Model) MetaValByIndex(i int) (string, error) {
	return adapters.ModelMetaValByIndex(m.ptr, i)
}

// Desc returns a string description of the model type.
func (m *Model) Desc() (string, error) {
	return adapters.ModelDesc(m.ptr)
}

func (m *Model) Size() uint64 {
	return adapters.ModelSize(m.ptr)
}

// ChatTemplate returns the chat template for the model. If name is empty, returns the default chat template.
func (m *Model) ChatTemplate(name string) (string, error) {
	return adapters.ModelChatTemplate(m.ptr, name)
}

func (m *Model) NumParams() uint64 {
	return adapters.ModelNumParams(m.ptr)
}

func (m *Model) HasEncoder() bool {
	return adapters.ModelHasEncoder(m.ptr)
}

func (m *Model) HasDecoder() bool {
	return adapters.ModelHasDecoder(m.ptr)
}

// DecoderStartToken returns the token that must be provided to the decoder to start generation in encoder-decoder models.
func (m *Model) DecoderStartToken() Token {
	return adapters.ModelDecoderStartToken(m.ptr)
}

func (m *Model) IsRecurrent() bool {
	return adapters.ModelIsRecurrent(m.ptr)
}

func (m *Model) IsHybrid() bool {
	return adapters.ModelIsHybrid(m.ptr)
}

func (m *Model) IsDiffusion() bool {
	return adapters.ModelIsDiffusion(m.ptr)
}

func (m *Model) InitAdapterLoRA(loraPath string) (Lora, error) {
	lora, err := adapters.InitAdapterLoRA(m.ptr, loraPath)
	return Lora{lora}, err
}

//
// Vocab
//

type Vocab struct {
	ptr *adapters.Vocab
}

func (v *Vocab) GetText(token Token) string {
	return adapters.GetVocabText(v.ptr, token)
}

func (v *Vocab) GetScore(token Token) float32 {
	return adapters.GetVocabScore(v.ptr, token)
}

func (v *Vocab) GetAttr(token Token) adapters.TokenAttr {
	return adapters.GetVocabAttr(v.ptr, token)
}

// IsEOG checks if the token is an end-of-generation token (e.g., EOS, EOT).
func (v *Vocab) IsEOG(token Token) bool {
	return adapters.VocabIsEOG(v.ptr, token)
}

// IsControl checks if the token is a control token (non-renderable).
func (v *Vocab) IsControl(token Token) bool {
	return adapters.VocabIsControl(v.ptr, token)
}

// BOS returns the beginning-of-sentence token.
func (v *Vocab) BOS() Token {
	return adapters.VocabBOS(v.ptr)
}

// EOS returns the end-of-sentence token.
func (v *Vocab) EOS() Token {
	return adapters.VocabEOS(v.ptr)
}

// EOT returns the end-of-turn token.
func (v *Vocab) EOT() Token {
	return adapters.VocabEOT(v.ptr)
}

// Sep returns the sentence separator token.
func (v *Vocab) Sep() Token {
	return adapters.VocabSep(v.ptr)
}

// NL returns the next-line token.
func (v *Vocab) NL() Token {
	return adapters.VocabNL(v.ptr)
}

// Pad returns the padding token.
func (v *Vocab) Pad() Token {
	return adapters.VocabPad(v.ptr)
}

// Mask returns the mask token.
func (v *Vocab) Mask() Token {
	return adapters.VocabMask(v.ptr)
}

// GetAddBOS returns whether BOS tokens are automatically added.
func (v *Vocab) GetAddBOS() bool {
	return adapters.GetAddBOS(v.ptr)
}

// GetAddEOS returns whether EOS tokens are automatically added.
func (v *Vocab) GetAddEOS() bool {
	return adapters.GetAddEOS(v.ptr)
}

// GetAddSep returns whether separator tokens are automatically added.
func (v *Vocab) GetAddSep() bool {
	return adapters.GetAddSep(v.ptr)
}

// FimPre returns the fill-in-the-middle prefix token.
func (v *Vocab) FimPre() Token {
	return adapters.VocabFimPre(v.ptr)
}

// FimSuf returns the fill-in-the-middle suffix token.
func (v *Vocab) FimSuf() Token {
	return adapters.VocabFimSuf(v.ptr)
}

// FimMid returns the fill-in-the-middle middle token.
func (v *Vocab) FimMid() Token {
	return adapters.VocabFimMid(v.ptr)
}

// FimPad returns the fill-in-the-middle padding token.
func (v *Vocab) FimPad() Token {
	return adapters.VocabFimPad(v.ptr)
}

// FimRep returns the fill-in-the-middle repeat token.
func (v *Vocab) FimRep() Token {
	return adapters.VocabFimRep(v.ptr)
}

// FimSep returns the fill-in-the-middle separator token.
func (v *Vocab) FimSep() Token {
	return adapters.VocabFimSep(v.ptr)
}

func (v *Vocab) Tokenize(text string, addSpecial bool, parseSpecial bool) ([]Token, error) {
	return adapters.Tokenize(v.ptr, text, addSpecial, parseSpecial)
}

// TokenToPiece converts a token ID to its string representation. lstrip skips leading spaces, special determines whether to render special tokens.
func (v *Vocab) TokenToPiece(token Token, lstrip int32, special bool) (string, error) {
	return adapters.TokenToPiece(v.ptr, token, lstrip, special)
}

// Detokenize converts a slice of tokens back into text. removeSpecial removes BOS/EOS if configured, unparseSpecial renders special tokens.
func (v *Vocab) Detokenize(tokens []Token, textLenMax int32, removeSpecial bool, unparseSpecial bool) (string, error) {
	return adapters.Detokenize(v.ptr, tokens, textLenMax, removeSpecial, unparseSpecial)
}

//
// Sampler
//

type Sampler struct {
	ptr *adapters.Sampler
}

func NewSamplerChain(params adapters.SamplerChainParams) Sampler {
	ptr := adapters.SamplerChainInit(params)
	return Sampler{ptr}
}

func (s *Sampler) Name() string {
	return adapters.SamplerName(s.ptr)
}

func (s *Sampler) Accept(token Token) {
	adapters.SamplerAccept(s.ptr, token)
}

func (s *Sampler) Apply(cur_p *adapters.TokenDataArray) {
	adapters.SamplerApply(s.ptr, cur_p)
}

func (s *Sampler) Reset() {
	adapters.SamplerReset(s.ptr)
}

func (s *Sampler) Clone() *Sampler {
	return &Sampler{ptr: adapters.SamplerClone(s.ptr)}
}

func (s *Sampler) Free() {
	adapters.SamplerFree(s.ptr)
}

// ChainAdd adds a sampler to a sampler chain (chain takes ownership and will free it).
func (s *Sampler) ChainAdd(smpl *Sampler) {
	adapters.SamplerChainAdd(s.ptr, smpl.ptr)
}

// ChainGet retrieves a sampler at the specified index from the chain.
func (s *Sampler) ChainGet(i int32) *Sampler {
	return &Sampler{ptr: adapters.SamplerChainGet(s.ptr, i)}
}

// ChainN returns the number of samplers in the chain.
func (s *Sampler) ChainN() int {
	return adapters.SamplerChainN(s.ptr)
}

// ChainRemove removes and returns a sampler from the chain at the specified index (chain no longer owns it).
func (s *Sampler) ChainRemove(i int32) *Sampler {
	return &Sampler{ptr: adapters.SamplerChainRemove(s.ptr, i)}
}

// GetSeed returns the seed used by the sampler (or default seed if not applicable).
func (s *Sampler) GetSeed() uint32 {
	return adapters.SamplerGetSeed(s.ptr)
}

func (s *Sampler) Sample(ctx *Context, idx int32) Token {
	return adapters.SamplerSample(s.ptr, ctx.ptr, idx)
}

//
// Sampler Initializers
//

func NewSamplerGreedy() *Sampler {
	return &Sampler{ptr: adapters.SamplerInitGreedy()}
}

func NewSamplerDist(seed uint32) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitDist(seed)}
}

func NewSamplerTopK(k int32) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitTopK(k)}
}

func NewSamplerTopP(p float32, minKeep uint) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitTopP(p, minKeep)}
}

func NewSamplerMinP(p float32, minKeep uint) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitMinP(p, minKeep)}
}

func NewSamplerTypical(p float32, minKeep uint) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitTypical(p, minKeep)}
}

func NewSamplerTemp(t float32) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitTemp(t)}
}

func NewSamplerTempExt(t float32, delta float32, exponent float32) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitTempExt(t, delta, exponent)}
}

func NewSamplerXTC(p float32, t float32, minKeep uint, seed uint32) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitXTC(p, t, minKeep, seed)}
}

func NewSamplerTopNSigma(n float32) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitTopNSigma(n)}
}

func NewSamplerMirostat(nVocab int32, seed uint32, tau float32, eta float32, m int32) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitMirostat(nVocab, seed, tau, eta, m)}
}

func NewSamplerMirostatV2(seed uint32, tau float32, eta float32) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitMirostatV2(seed, tau, eta)}
}

func NewSamplerGrammar(vocab *Vocab, grammarStr string, grammarRoot string) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitGrammar(vocab.ptr, grammarStr, grammarRoot)}
}

func NewSamplerGrammarLazyPatterns(vocab *Vocab, grammarStr string, grammarRoot string, triggerPatterns []string, triggerTokens []Token) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitGrammarLazyPatterns(vocab.ptr, grammarStr, grammarRoot, triggerPatterns, triggerTokens)}
}

func NewSamplerPenalties(penaltyLastN int32, penaltyRepeat float32, penaltyFreq float32, penaltyPresent float32) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitPenalties(penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent)}
}

func NewSamplerDry(vocab *Vocab, nCtxTrain int32, dryMultiplier float32, dryBase float32, dryAllowedLength int32, dryPenaltyLastN int32, seqBreakers []string) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitDry(vocab.ptr, nCtxTrain, dryMultiplier, dryBase, dryAllowedLength, dryPenaltyLastN, seqBreakers)}
}

func NewSamplerLogitBias(nVocab int32, nLogitBias int32, logitBias *adapters.LogitBias) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitLogitBias(nVocab, nLogitBias, logitBias)}
}

func NewSamplerInfill(vocab *Vocab) *Sampler {
	return &Sampler{ptr: adapters.SamplerInitInfill(vocab.ptr)}
}

//
// Sampler Chain Helpers
//

func (s *Sampler) AddSamplerGreedy() {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitGreedy())
}

func (s *Sampler) AddSamplerDist(seed uint32) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitDist(seed))
}

func (s *Sampler) AddSamplerTopK(k int32) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitTopK(k))
}

func (s *Sampler) AddSamplerTopP(p float32, minKeep uint) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitTopP(p, minKeep))
}

func (s *Sampler) AddSamplerMinP(p float32, minKeep uint) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitMinP(p, minKeep))
}

func (s *Sampler) AddSamplerTypical(p float32, minKeep uint) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitTypical(p, minKeep))
}

func (s *Sampler) AddSamplerTemp(t float32) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitTemp(t))
}

func (s *Sampler) AddSamplerTempExt(t float32, delta float32, exponent float32) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitTempExt(t, delta, exponent))
}

func (s *Sampler) AddSamplerXTC(p float32, t float32, minKeep uint, seed uint32) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitXTC(p, t, minKeep, seed))
}

func (s *Sampler) AddSamplerTopNSigma(n float32) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitTopNSigma(n))
}

func (s *Sampler) AddSamplerMirostat(nVocab int32, seed uint32, tau float32, eta float32, m int32) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitMirostat(nVocab, seed, tau, eta, m))
}

func (s *Sampler) AddSamplerMirostatV2(seed uint32, tau float32, eta float32) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitMirostatV2(seed, tau, eta))
}

func (s *Sampler) AddSamplerGrammar(vocab *Vocab, grammarStr string, grammarRoot string) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitGrammar(vocab.ptr, grammarStr, grammarRoot))
}

func (s *Sampler) AddSamplerGrammarLazyPatterns(vocab *Vocab, grammarStr string, grammarRoot string, triggerPatterns []string, triggerTokens []Token) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitGrammarLazyPatterns(vocab.ptr, grammarStr, grammarRoot, triggerPatterns, triggerTokens))
}

func (s *Sampler) AddSamplerPenalties(penaltyLastN int32, penaltyRepeat float32, penaltyFreq float32, penaltyPresent float32) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitPenalties(penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent))
}

func (s *Sampler) AddSamplerDry(vocab *Vocab, nCtxTrain int32, dryMultiplier float32, dryBase float32, dryAllowedLength int32, dryPenaltyLastN int32, seqBreakers []string) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitDry(vocab.ptr, nCtxTrain, dryMultiplier, dryBase, dryAllowedLength, dryPenaltyLastN, seqBreakers))
}

func (s *Sampler) AddSamplerLogitBias(nVocab int32, nLogitBias int32, logitBias *adapters.LogitBias) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitLogitBias(nVocab, nLogitBias, logitBias))
}

func (s *Sampler) AddSamplerInfill(vocab *Vocab) {
	adapters.SamplerChainAdd(s.ptr, adapters.SamplerInitInfill(vocab.ptr))
}

//
// Memory
//

type Memory struct {
	ptr adapters.MemoryT
}

// Clear clears the memory contents. If data is true, the data buffers are also cleared.
func (m *Memory) Clear(data bool) {
	adapters.ClearMemory(m.ptr, data)
}

// RemoveSequence removes all tokens in a sequence within the position range [p0, p1). Negative positions use default ranges.
func (m *Memory) RemoveSequence(seqId int32, p0 int32, p1 int32) {
	adapters.RemoveMemorySequence(m.ptr, seqId, p0, p1)
}

// CopySequence copies all tokens from one sequence to another within the position range [p0, p1).
func (m *Memory) CopySequence(seqIdSrc int32, seqIdDst int32, p0 int32, p1 int32) {
	adapters.CopyMemorySequence(m.ptr, seqIdSrc, seqIdDst, p0, p1)
}

// KeepSequence removes all tokens that do not belong to the specified sequence.
func (m *Memory) KeepSequence(seqId int32) {
	adapters.KeepMemorySequence(m.ptr, seqId)
}

// AddSequence adds a relative position delta to all tokens in a sequence within [p0, p1).
func (m *Memory) AddSequence(seqId int32, p0 int32, p1 int32, delta int32) {
	adapters.AddMemorySequence(m.ptr, seqId, p0, p1, delta)
}

// DivideSequence performs integer division of positions by a factor d (>1) for tokens in a sequence within [p0, p1).
func (m *Memory) DivideSequence(seqId int32, p0 int32, p1 int32, d int) {
	adapters.DivideMemorySequence(m.ptr, seqId, p0, p1, d)
}

// MinSequence returns the smallest position present in memory for a sequence (returns -1 if empty).
func (m *Memory) MinSequence(seqId int32) int32 {
	return adapters.MinMemorySequence(m.ptr, seqId)
}

// MaxSequence returns the largest position present in memory for a sequence (returns -1 if empty).
func (m *Memory) SequenceMax(seqId int32) int32 {
	return adapters.MemorySequenceMax(m.ptr, seqId)
}

// CanShift returns whether the memory supports position shifting.
func (m *Memory) CanShift() bool {
	return adapters.MemoryCanShift(m.ptr)
}

//
// LoRA
//

type Lora struct {
	ptr *adapters.AdapterLoRA
}

// MetaValStr retrieves a metadata value as a string by key name.
func (l *Lora) MetaValStr(key string) (string, error) {
	return adapters.AdapterMetaValStr(l.ptr, key)
}

// MetaCount returns the number of metadata key-value pairs.
func (l *Lora) MetaCount() int32 {
	return adapters.AdapterMetaCount(l.ptr)
}

// MetaKeyByIndex returns the metadata key name at the specified index.
func (l *Lora) MetaKeyByIndex(i int) (string, error) {
	return adapters.AdapterMetaKeyByIndex(l.ptr, i)
}

// MetaValByIndex returns the metadata value at the specified index.
func (l *Lora) MetaValByIndex(i int) (string, error) {
	return adapters.AdapterMetaValByIndex(l.ptr, i)
}

// Free manually frees a LoRA adapter. Note: loaded adapters are freed when the associated model is deleted.
func (l *Lora) Free() {
	adapters.AdapterLoRAFree(l.ptr)
}

// GetALoRANumInvocationTokens returns the number of invocation tokens if this is an ALORA adapter.
func (l *Lora) GetALoRANumInvocationTokens() uint64 {
	return adapters.AdapterGetALoRANumInvocationTokens(l.ptr)
}

// GetALoRAInvocationTokens returns the invocation tokens if this is an ALORA adapter.
func (l *Lora) GetALoRAInvocationTokens() ([]Token, error) {
	return adapters.AdapterGetALoRAInvocationTokens(l.ptr)
}
