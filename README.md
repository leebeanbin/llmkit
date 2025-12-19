# llmkit

**Unified toolkit for managing and using multiple LLM providers**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Features

- **🔄 Unified Interface**: Use OpenAI, Claude, Gemini, and Ollama with the same API
- **📊 Model Registry**: Auto-detect available models from your API keys
- **🎛️ Parameter Adaptation**: Automatically convert parameters for each provider
  - `max_tokens` → `max_completion_tokens` (OpenAI GPT-5)
  - `max_tokens` → `max_output_tokens` (Gemini)
  - `max_tokens` → `num_predict` (Ollama)
- **📦 Zero External Dependencies**: No `src.*` imports, fully independent
- **🔍 CLI Tools**: Inspect models and capabilities from command line
- **🚀 Pattern-Based Inference**: Auto-detect new model capabilities

---

## 📦 Installation

### Quick Start (Recommended) ⭐

```bash
pip install llmkit
```

**What's included by default:**
- ✅ Model registry and CLI tools
- ✅ **OpenAI** SDK (GPT-4, GPT-5, etc.)
- ✅ **Anthropic** SDK (Claude 3.5, etc.)

**Optional providers:**
- Gemini and Ollama are optional (see below)

This covers the most commonly used providers out of the box!

**After installation, see the welcome message:**
```bash
python -m llmkit.scripts.welcome
# or
python scripts/welcome.py
```

---

### Install Additional Providers

```bash
# Add Gemini support
pip install llmkit[gemini]

# Add Ollama support (local models)
pip install llmkit[ollama]

# Install all providers (Gemini + Ollama)
pip install llmkit[all]
```

---

### Installation Guide

| Command | OpenAI | Claude | Gemini | Ollama |
|---------|--------|--------|--------|--------|
| `pip install llmkit` | ✅ | ✅ | ❌ | ❌ |
| `pip install llmkit[gemini]` | ✅ | ✅ | ✅ | ❌ |
| `pip install llmkit[ollama]` | ✅ | ✅ | ❌ | ✅ |
| `pip install llmkit[all]` | ✅ | ✅ | ✅ | ✅ |

💡 **Tip:** If you try to use a provider without its SDK, llmkit will show you a helpful install message!

### Development Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llmkit.git
cd llmkit

# Install in editable mode with dev dependencies
pip install -e ".[dev,all]"
```

---

## 🚀 Quick Start

### 1. Set up environment variables

```bash
# .env file or export
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
export OLLAMA_HOST="http://localhost:11434"
```

### 2. Python Usage

```python
from llmkit import get_registry

# Get model registry
registry = get_registry()

# Check active providers
active_providers = registry.get_active_providers()
print(f"Active: {[p.name for p in active_providers]}")
# → Active: ['openai', 'ollama']

# Get available models
models = registry.get_available_models()
for model in models:
    print(f"{model.model_name} ({model.provider})")
# → gpt-4o (openai)
# → gpt-4o-mini (openai)
# → claude-3-5-sonnet-20241022 (anthropic)
# → ...

# Get model info
model_info = registry.get_model_info("gpt-4o-mini")
print(f"Streaming: {model_info.supports_streaming}")
print(f"Temperature: {model_info.supports_temperature}")
print(f"Max Tokens: {model_info.supports_max_tokens}")
```

### 3. CLI Usage

```bash
# List all available models
llmkit list

# Show specific model details
llmkit show gpt-4o-mini

# List active providers
llmkit providers

# Show summary
llmkit summary

# Export all model info as JSON
llmkit export > models.json
```

---

## 📚 Detailed Usage

### Model Registry

```python
from llmkit import get_registry

registry = get_registry()

# Get models by provider
openai_models = registry.get_available_models(provider="openai")
claude_models = registry.get_available_models(provider="anthropic")

# Get specific model info
model = registry.get_model_info("gpt-4o")
if model:
    print(f"Display Name: {model.display_name}")
    print(f"Max Tokens: {model.max_tokens}")
    print(f"Temperature Range: {model.default_temperature}")

    # Check parameter support
    for param in model.parameters:
        status = "✅" if param.supported else "❌"
        print(f"{status} {param.name}: {param.description}")
```

### Provider Information

```python
from llmkit import get_registry

registry = get_registry()

# Get all providers
providers = registry.get_all_providers()

for name, provider in providers.items():
    print(f"Provider: {name}")
    print(f"  Status: {provider.status.value}")
    print(f"  Env Key: {provider.env_key}")
    print(f"  Available: {provider.env_value_set}")
    print(f"  Models: {len(provider.available_models)}")
    print(f"  Default: {provider.default_model}")
```

### Using with Actual LLM Calls

```python
# Example with OpenAI (if you have openai installed)
from llmkit import get_registry

registry = get_registry()
model_info = registry.get_model_info("gpt-4o-mini")

# Get parameter configuration
params = {}
if model_info.supports_temperature:
    params["temperature"] = 0.7
if model_info.uses_max_completion_tokens:
    params["max_completion_tokens"] = 1000
elif model_info.supports_max_tokens:
    params["max_tokens"] = 1000

print(f"Using parameters: {params}")
# → Using parameters: {'temperature': 0.7, 'max_tokens': 1000}
```

---

## 🔍 CLI Commands

### `llmkit list`

List all available models with their capabilities.

```bash
$ llmkit list

활성화된 제공자: openai, ollama
총 모델 수: 25

✅ gpt-4o (openai)
   Streaming: True
   Temperature: True
   Max Tokens: True

✅ gpt-5-nano (openai)
   Streaming: True
   Temperature: False
   Max Tokens: False
   Uses max_completion_tokens: True

✅ phi3.5 (ollama)
   Streaming: True
   Temperature: True
   Max Tokens: True
```

### `llmkit show <model>`

Show detailed information about a specific model.

```bash
$ llmkit show gpt-4o-mini

모델: gpt-4o-mini
제공자: openai
설명: OpenAI의 빠르고 저렴한 모델

기능:
  - Streaming: ✅ Yes
  - Temperature: ✅ Yes (0.0-2.0)
  - Max Tokens: ✅ Yes (16384)

파라미터:
  ✅ temperature (float)
     기본값: 0.0
     필수: False
     설명: 응답의 창의성/랜덤성 조절

  ✅ max_tokens (int)
     기본값: 16384
     필수: False
     설명: 생성할 최대 토큰 수

사용 예제:
[... 코드 예시 ...]
```

### `llmkit providers`

Show all configured providers.

```bash
$ llmkit providers

제공자 목록:

✅ openai
   상태: active
   환경변수: OPENAI_API_KEY = 설정됨
   사용 가능한 모델: 15
   기본 모델: gpt-4o-mini

❌ anthropic
   상태: inactive
   환경변수: ANTHROPIC_API_KEY = 미설정
   사용 가능한 모델: 0

✅ ollama
   상태: active
   환경변수: OLLAMA_HOST = 설정됨
   사용 가능한 모델: 4
   기본 모델: qwen2.5:7b
```

### `llmkit summary`

Show quick summary.

```bash
$ llmkit summary

요약 정보:

총 제공자: 4
활성화된 제공자: 2
총 모델 수: 19

활성화된 제공자: openai, ollama
```

---

## 🎨 Model Information Structure

Each model provides detailed capability information:

```python
@dataclass
class ModelCapabilityInfo:
    model_name: str                    # "gpt-4o-mini"
    display_name: str                  # "GPT-4o Mini"
    provider: str                      # "openai"
    model_type: str                    # "llm"

    supports_streaming: bool           # True
    supports_temperature: bool         # True
    supports_max_tokens: bool          # True
    uses_max_completion_tokens: bool   # False (True for GPT-5)

    max_tokens: int                    # 16384
    default_temperature: float         # 0.0

    description: str                   # Model description
    use_case: str                      # Recommended use case
    parameters: List[ParameterInfo]    # Detailed parameter info
    example_usage: str                 # Code examples
```

---

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI
OPENAI_API_KEY="sk-..."

# Anthropic Claude
ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
GEMINI_API_KEY="..."

# Ollama (local)
OLLAMA_HOST="http://localhost:11434"
```

### Using .env File

```bash
# Create .env file in your project root
cat > .env << EOF
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
GEMINI_API_KEY=your-key
OLLAMA_HOST=http://localhost:11434
EOF
```

llmkit will automatically load from `.env` if `python-dotenv` is installed.

---

## 📖 Model Support

### OpenAI

- ✅ GPT-4o, GPT-4o-mini, GPT-4-turbo
- ✅ GPT-5, GPT-5-mini, GPT-5-nano (with `max_completion_tokens`)
- ✅ GPT-4.1 series (with `max_completion_tokens`)
- ✅ O3, O3-mini, O4-mini (reasoning models)
- ✅ Auto-detection of new models
- ✅ Date-versioned models (e.g., `gpt-5-nano-2025-08-07`)

### Anthropic Claude

- ✅ Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- ✅ Temperature range: 0.0-1.0 (auto-clamped)
- ✅ Date-versioned models (e.g., `claude-3-5-sonnet-20241022`)

### Google Gemini

- ✅ Gemini 2.5 Flash, Gemini 2.5 Pro
- ✅ Gemini 2.0, Gemini 1.5 series
- ✅ `max_output_tokens` parameter
- ✅ Thinking mode support (2.5+)

### Ollama

- ✅ All local models
- ✅ `num_predict` parameter
- ✅ Dynamic model detection

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llmkit --cov-report=html

# Run specific test
pytest tests/test_registry.py

# Run async tests
pytest tests/test_providers.py -v
```

---

## 🛠️ Development

```bash
# Install in editable mode
pip install -e ".[dev,all]"

# Format code
black llmkit tests

# Lint
ruff check llmkit

# Type check
mypy llmkit
```

---

## 📝 Examples

See [examples/](examples/) directory for more examples:

- [`basic_usage.py`](examples/basic_usage.py) - Basic registry usage
- [`check_providers.py`](examples/check_providers.py) - Check active providers
- [`model_params.py`](examples/model_params.py) - Get model parameters
- [`test_import.py`](examples/test_import.py) - Test package import

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

llmkit은 다음 프로젝트들에서 영감을 받았습니다:

### 참고한 프로젝트

- **[LangChain](https://github.com/langchain-ai/langchain)**: LLM 애플리케이션 개발 프레임워크의 선구자. 체인, 에이전트, 메모리 등의 개념을 참고했습니다.
- **[Claude (Anthropic)](https://www.anthropic.com/)**: 명확하고 간결한 코드 작성 철학의 영감을 받았습니다. 모토 "Claude Code"는 여기서 유래했습니다.
- **[TeddyNote](https://github.com/teddynote/teddynote)**: 터미널 UI 디자인과 사용자 경험에 대한 인사이트를 제공했습니다.

### 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 참고한 프로젝트들의 라이센스:
- LangChain: MIT License
- Claude API: Anthropic의 API 서비스 약관
- TeddyNote: 해당 프로젝트의 라이센스 정책

### 감사의 말

- OpenAI for GPT models
- Anthropic for Claude API
- Google for Gemini API
- Ollama team for local LLM support
- Rich library for beautiful terminal UI

---

## 📧 Contact

- Issues: https://github.com/yourusername/llmkit/issues
- Discussions: https://github.com/yourusername/llmkit/discussions

---

## 🗺️ Roadmap

- [ ] Automatic model metadata updates (LLM-assisted)
- [ ] Unified LLM interface (single API for all providers)
- [ ] Parameter adapter (auto-convert parameters)
- [ ] Model performance benchmarks
- [ ] Integration with LangChain
- [ ] Web dashboard

---

**Made with ❤️ for the LLM community**
