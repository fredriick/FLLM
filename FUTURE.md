# FLLM Future Implementations

## Features

- [x] **Web UI** - Browser-based chat interface (`fllm run <model> --web`)
- [x] **fllm scan** - Detect hardware and recommend models (DONE)
- [ ] **OpenAI API compatibility** - Drop-in replacement for OpenAI SDK
- [ ] **Multi-model routing** - Run different models for different tasks
- [ ] **Model conversion** - Convert HuggingFace models to GGUF directly
- [ ] **Prompt templates** - Library of system prompts for different use cases
- [x] **Streaming responses** - Real-time token streaming in CLI (via llama-cpp-python)
- [ ] **Docker support** - Containerized deployment
- [ ] **Remote inference** - Connect to remote FLLM servers
- [ ] **Usage metrics** - Track tokens/sec, latency, cost
- [ ] **Model comparison** - Benchmark multiple models side-by-side
- [ ] **Continuous batching** - vLLM-style high-throughput serving
- [ ] **Plugin system** - Custom backends or preprocessing
- [ ] **TTS/STT integration** - Voice input/output
- [ ] **RAG support** - Built-in document retrieval
- [ ] **Config files** - `~/.fllm/config.yaml` for defaults

## Coding CLI Support (AI coding tools)

- [x] **/v1/chat/completions** - OpenAI-compatible chat endpoint (via llama-cpp-python)
- [x] **Streaming support** - Server-Sent Events for streaming tokens
- [ ] **/v1/models endpoint** - List available models explicitly
- [ ] **API key bypass** - Allow local usage without API key
- [ ] **Alt-provider compatible format - Support additional native API formats
- [ ] **MCP server** - Model Context Protocol server implementation

## Infrastructure

- [ ] **Unit tests** - Test coverage for core modules
- [ ] **CI/CD** - GitHub Actions for testing
- [ ] **Documentation** - Full API documentation
- [ ] **Error handling** - Graceful error messages and recovery

---

*Generated: 2026-03-07*
