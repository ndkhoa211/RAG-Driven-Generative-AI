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
- `OPENROUTER_API_KEY` (Chapter 8)
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

### Chapter 4: Multimodal RAG
- Image/video frame extraction
- Object detection (YOLO models)
- Unified text+vision embedding space

**Critical Variable Flow:**
```python
# Text pipeline
documents_llm = [...]  # From ChromaDB data
vector_query_engine_llm = vector_store_index_llm.as_query_engine(...)
llm_response = vector_query_engine_llm.query(user_input)

# Image pipeline
documents = [...]  # From DeepLake image labels
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

### Chapter 6: Scaling with Pinecone
- Cloud-based vector database
- Production-grade indexing
- Three-phase pipeline: collection → indexing → generation

### Chapter 8: Dynamic RAG (Open-Source)
**Critical Migration:** Migrated from local Hugging Face models to OpenRouter cloud API to avoid 13.5GB model downloads and GPU requirements.

**Key Architecture:**
- Uses OpenRouter API instead of local models
- No model downloads required
- Free tier available (`gpt-oss-20b` model)
- OpenAI client library for clean API calls
- Manual HTTP requests method documented for learning
- SciQ dataset (10,481 Q&A pairs)

**Critical Variable Naming (Avoid Conflicts):**
```python
# ChromaDB client
import chromadb
client = chromadb.Client()  # Ephemeral (in-memory)

# OpenRouter API client (MUST use different variable name)
from openai import OpenAI
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
```

**LLaMA2 Function Pattern:**
```python
def LLaMA2(prompt):
    """Uses openai_client (not client) to avoid ChromaDB conflict"""
    response = openai_client.chat.completions.create(
        model=MODEL,  # e.g., "gpt-oss-20b"
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return [{'generated_text': response.choices[0].message.content}]
```

**ChromaDB Auto-Cleanup Pattern:**
```python
# Cells 26-28: Ensure fresh database on each run
client = chromadb.Client()
try:
    client.delete_collection(collection_name)
    print(f"✓ Deleted existing collection")
except:
    print(f"✓ No existing collection to delete")
```

**Cell Execution Order (Critical):**
1. Cell 8: Define MODEL variable
2. Cell 11: Initialize openai_client (uses MODEL)
3. Cell 51: Define LLaMA2 function (uses openai_client)
4. Cell 54+: Use LLaMA2 function

**If you change MODEL in cell 8, you MUST re-run cell 11!**

**Performance Notes:**
- Embedding: ~930 seconds (acceptable)
- Querying: ~840 seconds (acceptable)
- Evaluation (cell 44): 800+ minutes (OPTIONAL - can skip)
- Generation: 2-5 seconds per query

**Batch Processing Limits:**
```python
# ChromaDB max batch size: 5461
batch_size = 5000  # Safe margin
for batch_idx in range(total_batches):
    collection.add(ids=[...], documents=[...], metadatas=[...])
```

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

### Variable Not Defined Errors
**Cell Execution Order Critical:**
- Chapter 8: Must run cells in order: 8 (MODEL) → 11 (openai_client) → 51 (LLaMA2) → 54+ (usage)
- If you change MODEL in cell 8, re-run cell 11 to update openai_client context
- Always run data collection before embedding, embedding before generation

### Long-Running Cells
**Chapter 8 Cell 44 (Evaluation):**
- Takes 800+ minutes to process 10,481 similarity calculations
- This cell is OPTIONAL - only measures retrieval quality
- Can safely skip and continue to cell 47 for RAG generation
- If stuck, interrupt kernel and skip this cell

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

**Chapter 8 Processing Times:**
- Embedding 10,481 docs: ~930 seconds (15.5 min)
- Querying 10,481 questions: ~840 seconds (14 min)
- Similarity evaluation: 800+ minutes (OPTIONAL - can skip)

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

3. **Batch Processing:** 100 chunks per embedding batch (balance API rate limits vs. speed); 5000 for ChromaDB inserts

4. **Model Progression:** Start with GPT-4o (balanced), upgrade to reasoning models (o1, o3) only when needed for complex logic

5. **Evaluation Strategy:** Dual metrics (TF-IDF + embeddings) to catch both keyword and semantic relevance

6. **Chapter 8 Cloud API:** Use OpenRouter instead of local models to avoid 13.5GB downloads and GPU requirements

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
5. Implement Chapter 8 pattern (OpenRouter with free models)

**Typical Costs (Chapter 2 example, 1145 chunks):**
- Embedding generation: ~$0.03 (one-time)
- Query embeddings: ~$0.0001 per query
- GPT-4o generation: ~$0.01-0.05 per response

**Chapter 8 OpenRouter Costs:**
- Free models: `gpt-oss-20b`, `mistralai/mistral-7b-instruct:free`, `google/gemma-2-9b-it:free`
- Paid models: ~$0.00001-0.00007 per 1K tokens (very cheap)

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
- Batch processing for robustness (100-5000 docs/batch)
- Retry mechanisms for API failures
- Multiple retrieval strategy fallbacks (Chapter 1 Modular RAG)
- Configurable timeouts

## Known Issues & Workarounds

**Issue:** DeepLake Windows compatibility (v4.4.0+)
**Workaround:** Use `deeplake<4.0` (v3.9.52) specified in `pyproject.toml`

**Issue:** ChromaDB collection already exists
**Workaround:** Auto-delete pattern - recreates collection on each run

**Issue:** Markdown not rendering in notebooks
**Workaround:** Use `display(Markdown(text))` from `IPython.display`, not `markdown.markdown()`

**Issue:** ipywidgets warning for progress bars
**Workaround:** Install `ipywidgets` (included in dependencies)

**Issue:** Chapter 5 `fetch_and_clean()` AttributeError
**Workaround:** Fixed with null checks and comprehensive error handling

**Issue:** Pydantic validation warnings with LlamaIndex 0.10.x
**Workaround:** Downgrade to `pydantic==2.7.4` or wait for LlamaIndex update

**Issue:** Chapter 8 - Cell 44 taking 800+ minutes
**Cause:** Similarity evaluation on 10,481 documents using spaCy is extremely slow
**Workaround:** Skip cell 44 (evaluation) - it's optional for measuring retrieval quality

**Issue:** Chapter 8 - Naming conflict between ChromaDB client and OpenAI client
**Workaround:** Use `client` for ChromaDB and `openai_client` for OpenRouter

**Issue:** Chapter 8 - Empty responses from LLM
**Cause:** MODEL variable changed but openai_client not re-initialized
**Workaround:** Always re-run cell 11 after changing MODEL in cell 8

**Issue:** Chapter 8 - Variable not defined errors
**Cause:** Cells run out of order
**Workaround:** Always run: 8 → 11 → 26-29 → 35 → 40 → 51 → 54+

## Notebook Editing Best Practices

**Critical Variables to Track:**
- Always verify that key variables are defined before usage
- Common missing variables: `ind`, `documents`, query engines, response objects
- Use Python scripts for complex notebook editing (JSON manipulation)

**Variable Naming Conventions:**
- `vector_query_engine_llm` - Query engine for text-based LLM RAG
- `vector_query_engine` - Query engine for multimodal RAG
- `multimodal_response` - Saved response before variable overwriting
- `response` - Generic query response (often gets overwritten)
- `documents_llm` - LlamaIndex Documents from text data
- `documents` - Documents from image/multimodal data
- `client` - ChromaDB client (Chapter 8)
- `openai_client` - OpenRouter API client (Chapter 8)

**Cell Dependencies:**
- Cell order matters: define before use
- Save critical responses immediately after generation
- Import statements should precede usage
