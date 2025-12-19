# 🚀 llmkit

**Production-ready LLM toolkit with unified interface for multiple providers**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/github/stars/leebeanbin/llmkit?style=social)](https://github.com/leebeanbin/llmkit)

**llmkit** is a comprehensive, production-ready toolkit for building LLM applications with a unified interface across OpenAI, Anthropic, Google, and Ollama. Write once, run everywhere.

---

## ✨ Key Features

### 🎯 **Core Features**
- 🔄 **Unified Interface** - Single API for OpenAI, Anthropic, Google, Ollama
- 🎛️ **Intelligent Adaptation** - Automatic parameter conversion between providers
- 📊 **Model Registry** - Auto-detect available models from API keys
- 🔍 **CLI Tools** - Inspect models and capabilities from command line
- 💰 **Cost Tracking** - Accurate token counting and cost estimation

### 🏗️ **RAG & Document Processing**
- 📄 **Document Loaders** - PDF, CSV, TXT with automatic format detection
- ✂️ **Smart Text Splitters** - Semantic chunking with tiktoken
- 🔍 **Vector Search** - Chroma, FAISS, Pinecone, Qdrant, Weaviate
- 🎯 **RAG Pipeline** - Complete question-answering system in one line
- 🐛 **RAG Debugging** - Comprehensive debugging toolkit

### 🤖 **Advanced LLM Features**
- 🛠️ **Tools & Agents** - Function calling with ReAct pattern
- 🧠 **Memory Systems** - Buffer, window, token-based, summary memory
- ⛓️ **Chains** - Sequential, parallel, and custom chain composition
- 📊 **Output Parsers** - Pydantic, JSON, datetime, enum parsing
- 🔁 **Streaming** - Real-time response streaming with stats

### 📈 **Graph & Multi-Agent**
- 🕸️ **Graph Workflows** - LangGraph-style DAG execution
- 🤝 **Multi-Agent** - Sequential, parallel, hierarchical, debate patterns
- 🔄 **State Management** - Automatic state threading and checkpoints
- 📞 **Communication** - Inter-agent message passing

### 🎨 **Multimodal AI**
- 🖼️ **Vision RAG** - Image-based question answering with CLIP
- 🎙️ **Audio Processing** - Whisper STT, multi-provider TTS
- 🔊 **Audio RAG** - Search and QA across audio files
- 🌐 **Web Search** - Google, Bing, DuckDuckGo integration
- 🧮 **ML Integration** - TensorFlow, PyTorch, Scikit-learn

### 🏭 **Production Features**
- 💵 **Token & Cost** - tiktoken-based accurate counting, cost optimization
- 📝 **Prompt Templates** - Few-shot, chat, chain-of-thought templates
- 📊 **Evaluation** - BLEU, ROUGE, LLM-as-Judge, RAG metrics
- 🎯 **Fine-tuning** - OpenAI fine-tuning API integration
- 🛡️ **Error Handling** - Retry, circuit breaker, rate limiting
- 📈 **Tracing** - Distributed tracing with OpenTelemetry export

---

## 📦 Installation

### Quick Start

```bash
pip install llmkit
```

**Included by default:**
- ✅ OpenAI SDK (GPT-4o, o1, etc.)
- ✅ Anthropic SDK (Claude 3.5, etc.)

### Optional Providers

```bash
# Add Gemini support
pip install llmkit[gemini]

# Add Ollama support (local models)
pip install llmkit[ollama]

# Install all providers
pip install llmkit[all]

# Development installation
pip install llmkit[dev,all]
```

---

## 🚀 Quick Start

### Environment Setup

```bash
# Create .env file
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
export OLLAMA_HOST="http://localhost:11434"
```

### Basic Usage

```python
from llmkit import Client

# Unified interface - works with any provider
client = Client(model="gpt-4o")
response = client.chat("Explain quantum computing in simple terms")
print(response.content)

# Switch providers seamlessly
client = Client(model="claude-3-5-sonnet-20241022")
response = client.chat("Same question, different provider")

# Streaming
for chunk in client.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### RAG in One Line

```python
from llmkit import RAGChain

# Create RAG system from documents
rag = RAGChain.from_documents("docs/")

# Ask questions
answer = rag.query("What is this document about?")
print(answer)

# With sources
answer, sources = rag.query("Explain the main concept", include_sources=True)
for source in sources:
    print(f"Source: {source.document.metadata['source']}")
```

### Cost Optimization

```python
from llmkit import count_tokens, estimate_cost, get_cheapest_model

# Count tokens
tokens = count_tokens("Your text here", model="gpt-4o")
print(f"Tokens: {tokens}")

# Estimate cost
cost = estimate_cost(
    input_text="Your prompt",
    output_text="Expected response",
    model="gpt-4o"
)
print(f"Cost: ${cost.total_cost:.4f}")

# Find cheapest model
cheapest = get_cheapest_model(
    input_text="Your prompt",
    output_tokens=1000,
    models=["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"]
)
print(f"Use: {cheapest}")
```

---

## 📚 Core Modules

### 1. Client & Adapters

Unified interface with automatic parameter adaptation:

```python
from llmkit import Client, adapt_parameters

# Works across all providers
client = Client(model="gpt-4o")

# Parameters automatically adapted
response = client.chat(
    "Hello",
    temperature=0.7,
    max_tokens=1000,  # → max_completion_tokens for GPT-5
                       # → max_output_tokens for Gemini
                       # → num_predict for Ollama
)
```

### 2. Document Processing

```python
from llmkit import DocumentLoader, RecursiveCharacterTextSplitter

# Load documents
docs = DocumentLoader.load("docs/")  # PDF, CSV, TXT

# Smart splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " "]
)
chunks = splitter.split_documents(docs)
```

### 3. Embeddings & Vector Stores

```python
from llmkit import OpenAIEmbedding, ChromaVectorStore

# Create embeddings
embedding = OpenAIEmbedding(model="text-embedding-3-small")

# Vector store
store = ChromaVectorStore.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_db"
)

# Search
results = store.similarity_search("query", k=5)

# MMR search (diversity)
diverse_results = store.mmr_search("query", k=5, lambda_mult=0.5)
```

### 4. Tools & Agents

```python
from llmkit import Agent, Tool

# Define tools
@Tool.from_function
def calculator(expression: str) -> str:
    """Evaluate a math expression"""
    return str(eval(expression))

@Tool.from_function
def search(query: str) -> str:
    """Search the web"""
    # ... web search logic
    return results

# Create agent
agent = Agent(
    llm=client,
    tools=[calculator, search],
    max_iterations=10
)

# Run agent
result = agent.run("What is 25 * 17? Then search for that number in math history")
print(result.output)
```

### 5. Memory & Chains

```python
from llmkit import BufferMemory, SequentialChain, PromptChain

# Memory
memory = BufferMemory(max_messages=10)
memory.add_message("user", "Hello")
memory.add_message("assistant", "Hi there!")

# Chains
analyze_chain = PromptChain(
    llm=client,
    template="Analyze this text: {text}"
)

summarize_chain = PromptChain(
    llm=client,
    template="Summarize: {analysis}"
)

# Sequential execution
chain = SequentialChain(steps=[analyze_chain, summarize_chain])
result = chain.run(text="Long article...")
```

### 6. Graph Workflows

```python
from llmkit import StateGraph

# Create graph
graph = StateGraph()

def analyze(state):
    state["analysis"] = client.chat(f"Analyze: {state['input']}")
    return state

def decide(state):
    score = float(state["analysis"].split("Score:")[1])
    return "good" if score > 0.8 else "bad"

def improve(state):
    state["output"] = client.chat(f"Improve: {state['input']}")
    return state

# Build graph
graph.add_node("analyze", analyze)
graph.add_node("improve", improve)
graph.add_conditional_edges("analyze", decide, {
    "good": "END",
    "bad": "improve"
})

# Run
result = graph.compile().invoke({"input": "Draft text"})
```

### 7. Multi-Agent Systems

```python
from llmkit import MultiAgentCoordinator, DebateStrategy

# Create agents
researcher = Agent(llm=client, tools=[search], role="researcher")
writer = Agent(llm=client, role="writer")
critic = Agent(llm=client, role="critic")

# Coordinate with debate
coordinator = MultiAgentCoordinator(
    agents=[researcher, writer, critic],
    strategy=DebateStrategy(rounds=3)
)

result = coordinator.coordinate("Write an article about quantum computing")
print(result.final_output)
```

### 8. Vision RAG

```python
from llmkit import VisionRAG, CLIPEmbedding, ImageLoader

# Load images
images = ImageLoader.load("images/")

# Create vision RAG
vision_rag = VisionRAG.from_images(
    images=images,
    embedding=CLIPEmbedding(),
    llm=Client(model="gpt-4o")  # Vision-capable model
)

# Query with text
answer = vision_rag.query("What objects are in these images?")

# Query with image
answer = vision_rag.query_with_image(
    "reference.jpg",
    "Find similar images and describe them"
)
```

### 9. Audio Processing

```python
from llmkit import WhisperSTT, TextToSpeech, AudioRAG

# Speech to text
stt = WhisperSTT()
result = stt.transcribe("audio.mp3", language="en")
print(result.text)

# Text to speech
tts = TextToSpeech(provider="openai")
audio = tts.synthesize("Hello world", voice="alloy", speed=1.0)

# Audio RAG
audio_rag = AudioRAG.from_audio_files([
    "podcast1.mp3",
    "podcast2.mp3"
])
answer = audio_rag.query("What was discussed about AI?")
```

### 10. Web Search

```python
from llmkit import DuckDuckGoSearch, WebScraper

# Search (no API key needed!)
search = DuckDuckGoSearch()
results = search.search("latest AI news", max_results=5)

for result in results:
    print(f"{result.title}: {result.url}")

# Scrape content
scraper = WebScraper()
content = scraper.scrape(results[0].url)
print(content)
```

### 11. Prompt Templates

```python
from llmkit import PromptTemplate, FewShotPromptTemplate, PredefinedTemplates

# Basic template
template = PromptTemplate(
    template="Translate {text} from {source} to {target}",
    input_variables=["text", "source", "target"]
)
prompt = template.format(text="Hello", source="English", target="Korean")

# Few-shot template
from llmkit import PromptExample

examples = [
    PromptExample(input="2+2", output="4"),
    PromptExample(input="3*5", output="15")
]

few_shot = FewShotPromptTemplate(
    examples=examples,
    example_template=PromptTemplate(
        template="Q: {input}\nA: {output}",
        input_variables=["input", "output"]
    ),
    prefix="Solve the math problem:",
    suffix="Q: {input}\nA:"
)

# Predefined templates
cot = PredefinedTemplates.chain_of_thought()
prompt = cot.format(question="What is 25% of 80?")
```

### 12. Evaluation

```python
from llmkit import BLEUMetric, ROUGEMetric, evaluate_text, evaluate_rag

# Text evaluation
prediction = "The cat sits on the mat"
reference = "The cat is sitting on the mat"

result = evaluate_text(
    prediction=prediction,
    reference=reference,
    metrics=["bleu", "rouge-1", "rouge-l", "f1"]
)
print(f"Average score: {result.average_score:.4f}")

# RAG evaluation
rag_result = evaluate_rag(
    question="What is AI?",
    answer="AI is artificial intelligence...",
    contexts=["Context 1", "Context 2"],
    ground_truth="AI is..."
)
```

### 13. Fine-tuning

```python
from llmkit import DatasetBuilder, FineTuningManager, create_finetuning_provider

# Prepare data
qa_pairs = [
    {"question": "What is Python?", "answer": "Python is..."},
    {"question": "What is a list?", "answer": "A list is..."}
]

examples = DatasetBuilder.from_qa_pairs(
    qa_pairs,
    system_message="You are a Python expert"
)

# Split data
train, val = DatasetBuilder.split_dataset(examples, train_ratio=0.8)

# Fine-tune
provider = create_finetuning_provider("openai")
manager = FineTuningManager(provider)

train_file = manager.prepare_and_upload(train, "train.jsonl")
val_file = manager.prepare_and_upload(val, "val.jsonl")

job = manager.start_training(
    model="gpt-3.5-turbo",
    training_file=train_file,
    validation_file=val_file,
    n_epochs=3
)
```

### 14. Error Handling

```python
from llmkit import retry, circuit_breaker, rate_limit, with_error_handling

# Retry with exponential backoff
@retry(max_retries=3, strategy=RetryStrategy.EXPONENTIAL)
def api_call():
    return client.chat("Hello")

# Circuit breaker
@circuit_breaker(failure_threshold=5, timeout=60)
def flaky_service():
    return external_api.call()

# Rate limiting
@rate_limit(max_calls=10, time_window=60)
def rate_limited_call():
    return api.call()

# Combined error handling
@with_error_handling(max_retries=3, failure_threshold=5, max_calls=10)
def production_call():
    return client.chat("Production query")
```

---

## 🎓 Documentation & Learning

### Complete Learning Path

llmkit includes **comprehensive AI master's level documentation**:

- **Theory Documents** (900+ lines each with mathematical proofs)
  - Embeddings: Vector math, Word2Vec, Attention, Transformers
  - RAG: Vector search, HNSW, MMR, hybrid search
  - Graph Workflows: DAG, topological sort, Petri nets
  - Multi-Agent: Game theory, consensus, debate
  - Vision: CNN, ResNet, CLIP, vision transformers
  - Audio: Nyquist, MFCC, CTC, Whisper, WaveNet
  - Production: Tokenization, BLEU/ROUGE, LoRA, error handling

- **Tutorials** (600+ lines each with practical examples)
  - Step-by-step implementations
  - Real-world use cases
  - Performance benchmarking

- **16-Week Curriculum** (`docs/LEARNING_PATH.md`)
  - Structured learning from basics to advanced
  - Projects and exercises
  - Graduate-level depth

See [`docs/`](docs/) directory for all materials.

---

## 🔧 CLI Usage

```bash
# List available models
llmkit list

# Show model details
llmkit show gpt-4o

# Check providers
llmkit providers

# Quick summary
llmkit summary

# Export model info
llmkit export > models.json
```

---

## 🌟 Examples

Check [`examples/`](examples/) directory:

- `basic_usage.py` - Getting started
- `rag_demo.py` - RAG system
- `agent_demo.py` - Tool-using agents
- `graph_demo.py` - Graph workflows
- `multi_agent_demo.py` - Multi-agent systems
- `vision_rag_demo.py` - Vision RAG
- `audio_demo.py` - Audio processing

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=llmkit --cov-report=html

# Specific module
pytest tests/test_rag.py -v
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

## 🗺️ Roadmap

- ✅ Unified multi-provider interface
- ✅ RAG pipeline
- ✅ Tools & Agents
- ✅ Graph workflows
- ✅ Multi-agent systems
- ✅ Vision & Audio
- ✅ Production features
- ⬜ LangSmith integration
- ⬜ Prompt optimization
- ⬜ Model benchmarks
- ⬜ Web dashboard

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Inspired by:
- **[LangChain](https://github.com/langchain-ai/langchain)** - LLM application framework
- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Graph workflow patterns
- **[Anthropic Claude](https://www.anthropic.com/)** - Clear code philosophy

Special thanks to:
- OpenAI for GPT models and APIs
- Anthropic for Claude API
- Google for Gemini API
- Ollama team for local LLM support

---

## 📧 Contact

- **GitHub**: https://github.com/leebeanbin/llmkit
- **Issues**: https://github.com/leebeanbin/llmkit/issues
- **Discussions**: https://github.com/leebeanbin/llmkit/discussions

---

**Built with ❤️ for the LLM community**

Transform your LLM applications from prototype to production with llmkit.
