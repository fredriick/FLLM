# FLLM Future Implementations

## Features

- [x] **Web UI** - Browser-based chat interface (`fllm run <model> --web`)
- [x] **fllm scan** - Detect hardware and recommend models (DONE)
- [x] **OpenAI API compatibility** - Drop-in replacement for OpenAI SDK (`fllm serve <model>`)
- [ ] **Multi-model routing** - Run different models for different tasks
- [ ] **Model conversion** - Convert HuggingFace models to GGUF directly
- [x] **Prompt templates** - Library of system prompts for different use cases (`fllm templates`)
- [x] **Streaming responses** - Real-time token streaming in CLI (via llama-cpp-python)
- [ ] **Docker support** - Containerized deployment
- [x] **Remote inference** - Connect to remote FLLM servers (`fllm remote`)
- [x] **Usage metrics** - Track tokens/sec, latency, cost (`fllm metrics`)
- [x] **Model comparison** - Benchmark multiple models side-by-side (`fllm compare`)
- [ ] **Continuous batching** - vLLM-style high-throughput serving
- [ ] **Plugin system** - Custom backends or preprocessing
- [ ] **TTS/STT integration** - Voice input/output
- [ ] **RAG support** - Built-in document retrieval
- [x] **Config files** - `~/.fllm/config.yaml` for defaults (`fllm config`)

## Coding CLI Support (AI coding tools)

- [x] **/v1/chat/completions** - OpenAI-compatible chat endpoint (via llama-cpp-python)
- [x] **Streaming support** - Server-Sent Events for streaming tokens
- [x] **/v1/models endpoint** - List available models explicitly
- [x] **API key bypass** - Allow local usage without API key
- [ ] **Alt-provider compatible format - Support additional native API formats
- [ ] **MCP server** - Model Context Protocol server implementation

## Infrastructure

- [ ] **Unit tests** - Test coverage for core modules
- [ ] **CI/CD** - GitHub Actions for testing
- [ ] **Documentation** - Full API documentation
- [x] **Error handling** - Graceful error messages and recovery

---

*Generated: 2026-03-07*
