# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the official code repository for "RAG-Driven Generative AI" (Packt Publishing), an educational resource for building enterprise-grade Retrieval Augmented Generation (RAG) pipelines. The project progresses from basic RAG concepts through production-scale multimodal implementations.

## Environment Setup

**Python Version:** 3.12+ (managed via `uv` package manager)

**Install Dependencies:**
```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -r requirements.txt  # if available
```

**Required API Keys (.env file):**
```
OPENAI_API_KEY=sk-proj-...
```

Optional for specific chapters:
- `ACTIVELOOP_TOKEN` (Chapters 3, 7 - Deep Lake)
- `PINECONE_API_KEY` (Chapter 6)
- `TAVILY_API_KEY`, `LANGCHAIN_API_KEY` (Advanced features)

**spaCy Models:**
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md  # if needed
```

## Running Notebooks

**Execute in Order per Chapter:**
```bash
# Chapter 2 example (typical 3-notebook flow):
jupyter notebook Chapter02/1_Data_collection_preparation.ipynb  # Step 1: Collect data
jupyter notebook Chapter02/2_Embeddings_vector_store.ipynb      # Step 2: Create embeddings
jupyter notebook Chapter02/3_Augmented_Generation.ipynb         # Step 3: RAG generation
```

**Key Configuration Variables (in notebooks):**
```python
CHUNK_SIZE = 1000           # Text chunking size
BATCH_SIZE = 100            # Embedding batch size
N_RESULTS = 10              # Top-K retrieval
GPT_MODEL = "gpt-4o"        # LLM selection
CHROMA_PATH = "./chroma_db" # Vector store path
```

## Architecture: RAG Pipeline Flow

The codebase implements a 5-phase RAG pipeline:

### Phase 1: Data Collection (Notebook 1 in each chapter)
- Web scraping with BeautifulSoup (Wikipedia, domain-specific sources)
- Text cleaning and normalization
- Output: Raw text files (`llm.txt`, dataset-specific)

### Phase 2: Embedding & Storage (Notebook 2)
- Fixed-size chunking (default 1000 chars)
- Batch embedding generation via OpenAI `text-embedding-3-small` (1536-dim)
- Vector store persistence (ChromaDB or Deep Lake)

### Phase 3: Retrieval (Core RAG component)
- Query embedding using same model
- Similarity search (Euclidean distance for ChromaDB)
- Top-K retrieval with metadata

### Phase 4: Augmented Generation (Notebook 3)
- Context concatenation (query + retrieved docs)
- LLM inference (GPT-4o, temperature 0.1 for factual)
- Response generation

### Phase 5: Evaluation
- TF-IDF cosine similarity (sklearn)
- Embedding-based cosine similarity (sentence-transformers)
- Performance metrics (latency, accuracy)

## Vector Store Architecture

**Two Primary Strategies:**

### ChromaDB (Chapters 2, 8 - Local Development)
```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_collection")

# Storage: Local SQLite + Parquet files
# Best for: Development, prototyping, dynamic RAG
# Cost: Free (no cloud dependencies)
```

### Deep Lake (Chapters 3, 7 - Production Scale)
```python
import deeplake
vector_store_path = "hub://organization/dataset_name"
ds = deeplake.load(vector_store_path)

# Storage: Activeloop cloud backend
# Best for: Enterprise scale, multimodal data
# Cost: Paid cloud infrastructure
```

**Key Difference:** ChromaDB is local-first and lightweight; Deep Lake requires cloud credentials but scales better for large datasets and supports multimodal (text + image).

## LLM Model Selection Guide

| Model | Use Case | Trade-offs |
|-------|----------|-----------|
| `gpt-4o` | Default production (Chapters 2-6) | Balanced performance/cost |
| `gpt-4.5-preview` | Agentic workflows (Chapter 2 bonus) | Tool use, multi-step reasoning |
| `o1-preview`, `o3` | Complex reasoning (Chapter 2 bonus) | Higher latency, better logic |
| `llama-2` | Open-source (Chapter 8) | No API costs, resource-intensive |
| `grok-beta` | xAI alternative (Chapter 1 bonus) | Proprietary alternative |

**Temperature Settings:**
- `0.1` for factual RAG responses (minimize hallucination)
- `0.7-0.9` for creative generation

## Chapter-Specific Patterns

### Chapter 1: RAG Foundations
- **Naive RAG:** Simple keyword search + LLM concatenation
- **Advanced RAG:** TF-IDF vectorization + cosine similarity ranking
- **Modular RAG:** Switchable retrieval strategies (keyword, vector, indexed)

### Chapter 2: Core Pipeline (ChromaDB)
**Critical Migration Note:** This chapter was migrated from DeepLake to ChromaDB for local development. All notebooks use:
- `dotenv` for API keys (not Google Colab)
- Local `./chroma_db` storage
- UTF-8 encoding for Windows compatibility
- Auto-refresh collection on each run (no duplicate handling)

**Typical Issues:**
- Missing `.env` file: Create with `OPENAI_API_KEY`
- ChromaDB not found: Run notebook 2 before notebook 3
- Encoding errors: Ensure `encoding='utf-8'` in file operations

### Chapter 3: Index-Based RAG (LlamaIndex)
Multiple index types with performance trade-offs:
```python
# VectorStoreIndex: Fastest (0.6312 perf score)
# TreeIndex: Moderate (0.1686)
# ListIndex: Comprehensive but slowest (0.0475)
# KeywordTableIndex: Keyword-optimized
```

### Chapters 4, 10: Multimodal RAG
- Image/video frame extraction
- Object detection (YOLO models)
- Unified text+vision embedding space

### Chapter 6: Scaling with Pinecone
- Cloud-based vector database
- Production-grade indexing
- Three-phase pipeline: collection → indexing → generation

### Chapter 5: Adaptive RAG (Human-in-the-Loop)
**Critical Migration Note:** This chapter was migrated from Google Colab to local Jupyter. All code now uses:
- `dotenv` for API keys (not Google Colab)
- `ipywidgets` for interactive feedback (not `google.colab.output`)
- UTF-8 encoding for Windows compatibility
- Icon images loaded from `../commons/` directory

**Adaptive RAG Strategy Selection:**
```python
# Three RAG strategies based on performance ranking (1-5):
ranking = 5  # User-configurable

# Ranking 1-2: No RAG (Direct LLM)
if ranking >= 1 and ranking < 3:
    text_input = user_input

# Ranking 3-4: Human Expert Feedback RAG only
if ranking >= 3 and ranking < 5:
    # Loads human_feedback.txt for domain-specific context
    text_input = human_feedback_content

# Ranking 5: Document RAG only
if ranking >= 5:
    # Wikipedia retrieval with fetch_and_clean()
    text_input = retrieved_documents
```

**Three-Phase Pipeline:**
1. **RETRIEVER:** Wikipedia scraping with robust error handling
2. **GENERATOR:** Adaptive strategy selection + GPT-4o generation
3. **EVALUATOR:** Multi-metric scoring (cosine similarity + human ratings)

**Interactive Feedback Interface:**
- Uses `ipywidgets` buttons (👍/👎) for expert evaluation
- Saves feedback to `expert_feedback.txt` for adaptive loop
- Fallback to text mode if icon images not found
- Tracks mean score history across queries

**Typical Issues:**
- Missing `.env` file: Create with `OPENAI_API_KEY`
- Wikipedia fetch errors: Fixed with comprehensive error handling in `fetch_and_clean()`
- Icon images not found: Place in `commons/` directory or use emoji fallback

### Chapter 8: Dynamic RAG (Open-Source)
- Hugging Face Llama-2 integration
- Cost-effective alternative to OpenAI
- Real-time dataset updates
- SciQ dataset (10,481 Q&A pairs)

## Embedding Strategy

**Development:** Sentence Transformers (`all-MiniLM-L6-v2`, 384-dim)
- Local inference, no API calls
- Good for experimentation

**Production:** OpenAI (`text-embedding-3-small`, 1536-dim)
- Superior semantic understanding
- Consistent with GPT models
- Cost: ~$0.00002 per 1K tokens

**Important:** Always use the same embedding model for indexing and querying.

## Common Debugging Patterns

### API Key Issues
```python
# Verify API key loaded correctly
import os
from dotenv import load_dotenv
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "API key not found in .env"
```

### ChromaDB Collection Not Found
```python
# Check existing collections
client = chromadb.PersistentClient(path="./chroma_db")
print(client.list_collections())

# Recreate if needed (notebook 2 pattern)
try:
    client.delete_collection(name="collection_name")
except:
    pass
collection = client.create_collection(name="collection_name")
```

### Embedding Dimension Mismatch
- Symptom: "Dimension mismatch" errors during query
- Cause: Different embedding models for indexing vs. querying
- Solution: Recreate vector store with consistent model

### Windows UTF-8 Encoding
```python
# Always specify encoding on Windows
with open('file.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

## Performance Metrics Reference

**Expected Values (Chapter 2 baseline):**
- TF-IDF Similarity: 0.4-0.8 range (RAG improves by ~0.3-0.4)
- Embedding Similarity: 0.6-0.9 range
- Query Latency: 2-5 seconds (GPT-4o)
- Distance Scores: 0.4-1.0 (lower is better, <0.6 = highly relevant)

**Chapter 3 Index Performance:**
```
Vector Index:  0.6312 (speed/accuracy balance)
Tree Index:    0.1686 (hierarchical summarization)
List Index:    0.0475 (most thorough, slowest)
```

## Data Storage Patterns

**Local Files (gitignored):**
- `./chroma_db/` - ChromaDB persistence
- `./.deeplake/` - Deep Lake local cache
- `*.pkl`, `*.h5` - Model checkpoints
- `data/` - Raw datasets

**Version Controlled:**
- Notebooks (`.ipynb`)
- Configuration (`pyproject.toml`, `.python-version`)
- Documentation (`README.md`, `CHANGELOG.md`)
- Shared assets (`commons/thumbs_up.png`, `commons/thumbs_down.png`)

## Critical Architectural Decisions

1. **ChromaDB vs. Deep Lake:** ChromaDB for Chapters 2, 8 (local dev); Deep Lake for Chapters 3, 7 (cloud scale)

2. **Fixed Chunking:** 1000-character chunks with no overlap (simple but may lose boundary context)

3. **Batch Processing:** 100 chunks per embedding batch (balance API rate limits vs. speed)

4. **Model Progression:** Start with GPT-4o (balanced), upgrade to reasoning models (o1, o3) only when needed for complex logic

5. **Evaluation Strategy:** Dual metrics (TF-IDF + embeddings) to catch both keyword and semantic relevance

## Chapter Dependency Graph

```
Chapter 1 (Foundations) → Chapters 2-10 (all build on RAG concepts)
    ↓
Chapter 2 (Core Pipeline) → Chapter 5 (Adaptive RAG) → Chapter 8 (Dynamic RAG)
    ↓
Chapter 3 (LlamaIndex) → Chapter 7 (Knowledge Graphs)
    ↓
Chapter 4 (Multimodal) → Chapter 10 (Video)
    ↓
Chapter 6 (Scaling) → Chapter 9 (Fine-tuning)
```

**Recommended Learning Path:** Chapters 1 → 2 → 5 → 3 → 8 for core RAG understanding (includes human-in-the-loop patterns), then specialized chapters as needed.

## Testing Retrieval Quality

**Quick validation after building vector store:**
```python
# Test query
test_query = "What is [domain-specific term]?"
results = collection.query(
    query_embeddings=[get_embeddings([test_query])[0]],
    n_results=5
)

# Check distances (should be <0.8 for relevant results)
for i, dist in enumerate(results['distances'][0]):
    print(f"Result {i+1}: distance={dist:.4f}")
    print(f"Text: {results['documents'][0][i][:100]}...")
```

**Expected:** Top result distance <0.6 indicates good retrieval quality.

## Cost Optimization

**Reduce OpenAI API Costs:**
1. Use sentence-transformers for embeddings (local, free)
2. Cache embeddings to avoid regeneration
3. Limit `N_RESULTS` to minimum needed (default 10, try 3-5)
4. Use GPT-4o-mini for non-critical generation (Chapter 9)
5. Implement Chapter 8 pattern (open-source Llama-2) for high-volume

**Typical Costs (Chapter 2 example, 1145 chunks):**
- Embedding generation: ~$0.03 (one-time)
- Query embeddings: ~$0.0001 per query
- GPT-4o generation: ~$0.01-0.05 per response

## Multimodal Extensions

**Image + Text RAG (Chapter 4):**
```python
# Pattern: Extract text captions from images
# Embed both text and image features
# Unified similarity search across modalities
```

**Video RAG (Chapter 10):**
```python
# Extract frames at intervals
# Apply object detection (YOLO)
# Generate metadata per frame
# Store in Pinecone for temporal search
```

## Production Deployment Considerations

**Single Machine → Scaled:**
- Dev: ChromaDB + OpenAI (Chapters 1-2)
- Production: Deep Lake/Pinecone + LlamaIndex (Chapters 3, 6)
- Enterprise: Multi-agent + fine-tuning (Chapters 9-10)

**Reliability Patterns:**
- Batch processing for robustness (100 docs/batch)
- Retry mechanisms for API failures
- Multiple retrieval strategy fallbacks (Chapter 1 Modular RAG)
- Configurable timeouts

## Known Issues & Workarounds

**Issue:** DeepLake Windows compatibility (v4.4.0+)
**Workaround:** Use `deeplake<4.0` (v3.9.52) specified in `pyproject.toml`

**Issue:** ChromaDB collection already exists
**Workaround:** Auto-delete pattern (Chapter 2, notebook 2) - recreates collection on each run

**Issue:** Markdown not rendering in notebooks
**Workaround:** Use `display(Markdown(text))` from `IPython.display`, not `markdown.markdown()`

**Issue:** ipywidgets warning for progress bars
**Workaround:** Install `ipywidgets` (included in dependencies since Chapter 5 update)

**Issue:** Chapter 5 `fetch_and_clean()` AttributeError: 'NoneType' object has no attribute 'find'
**Workaround:** Fixed in latest version with null checks and comprehensive error handling

**Issue:** Pydantic validation warnings with LlamaIndex 0.10.x
**Workaround:** Downgrade to `pydantic==2.7.4` or wait for LlamaIndex update

## Notebook Editing Best Practices

When editing Jupyter notebooks programmatically:

**Critical Variables to Track:**
- Always verify that key variables are defined before usage
- Common missing variables after cell deletion: `ind`, `documents`, query engines, response objects
- Use Python scripts to edit notebooks (JSON manipulation) for complex operations

**Variable Naming Conventions:**
- `vector_query_engine_llm` - Query engine for text-based LLM RAG
- `vector_query_engine` - Query engine for multimodal RAG
- `multimodal_response` - Saved response before variable overwriting
- `response` - Generic query response (often gets overwritten)
- `documents_llm` - LlamaIndex Documents from text data
- `documents` - Documents from image/multimodal data

**Cell Dependencies:**
- Cell order matters: define before use
- Save critical responses immediately after generation
- Import statements should precede usage (sklearn, PIL, etc.)

## Chapter 4 Specific Notes

### Multimodal RAG Architecture

**Two Parallel Pipelines:**
1. **Text Pipeline (LLM):** ChromaDB → LlamaIndex → GPT-4o text response
2. **Image Pipeline (Vision):** DeepLake → LlamaIndex → GPT-4o vision analysis

**Critical Variable Flow:**
```python
# Pipeline 1: Text RAG
documents_llm = [...]  # From ChromaDB data
vector_store_index_llm = VectorStoreIndex.from_documents(documents_llm)
vector_query_engine_llm = vector_store_index_llm.as_query_engine(...)
llm_response = vector_query_engine_llm.query(user_input)

# Pipeline 2: Multimodal RAG
documents = [...]  # From DeepLake image labels
vector_store_index = GPTVectorStoreIndex.from_documents(documents)
vector_query_engine = vector_store_index.as_query_engine(...)
response = vector_query_engine.query(user_input)
multimodal_response = response  # CRITICAL: Save before overwriting!
```

**Key Functions:**
- `get_unique_words()` - Extracts unique words from retrieved text
- `process_and_display()` - Matches query target to image dataset
- `display_image_with_bboxes()` - Renders images with object detection boxes
- `display_source_image()` - Shows saved images
- `calculate_cosine_similarity_with_embeddings()` - Performance evaluation

### DeepLake Integration Pattern
```python
# Load cloud dataset (read-only)
dataset_path = 'hub://activeloop/visdrone-det-train'
ds = deeplake.load(dataset_path)  # 6,471 images

# Convert to pandas DataFrame
df = pd.DataFrame(columns=['image', 'boxes', 'labels'])
for i, sample in enumerate(ds):
    df.loc[i, 'image'] = sample.images.tobytes()
    df.loc[i, 'boxes'] = [box.tolist() for box in sample.boxes.numpy(aslist=True)]
    df.loc[i, 'labels'] = sample.labels.data()['text']

# Add unique IDs for LlamaIndex
df['doc_id'] = df.index.astype(str)
```

### Intelligent Target Detection
The notebook uses query-aware object detection:
```python
# Extract target object from user query
query_words = user_input.lower().split()  # "How do drones identify a truck?"
target_word = None
for word in unique_words:  # ["truck", "car", "van"]
    if word in query_words:  # Matches "truck"
        target_word = word
        break
```

This ensures bounding boxes match the user's intent, not just the most frequent object.
