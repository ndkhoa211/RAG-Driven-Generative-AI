# Learning LlamaIndex: A Complete Workflow Guide

This guide explains how to use LlamaIndex for building RAG (Retrieval-Augmented Generation) systems, based on practical examples from Chapters 3 and 4.

---

## Table of Contents

1. [What is LlamaIndex?](#what-is-llamaindex)
2. [Installation & Setup](#installation--setup)
3. [Core Workflow: 5-Phase RAG Pipeline](#core-workflow-5-phase-rag-pipeline)
4. [Document Loading & Preparation](#document-loading--preparation)
5. [Index Types & When to Use Them](#index-types--when-to-use-them)
6. [Query Engines & Retrieval](#query-engines--retrieval)
7. [Vector Store Integration](#vector-store-integration)
8. [Advanced Patterns](#advanced-patterns)
9. [Best Practices](#best-practices)
10. [Common Patterns Comparison](#common-patterns-comparison)

---

## What is LlamaIndex?

**LlamaIndex** is a data framework for building LLM-powered applications. It orchestrates the entire RAG pipeline:

- **Document Loading**: Reads files from various sources
- **Chunking**: Splits documents into manageable pieces
- **Embedding**: Converts text to vector representations
- **Indexing**: Organizes data for efficient retrieval
- **Querying**: Combines retrieval + LLM generation

Think of it as the **"operating system"** for your RAG application, while vector databases like ChromaDB are the **"storage layer"**.

---

## Installation & Setup

### Required Packages

```bash
# Core LlamaIndex
uv add llama-index==0.10.64

# Vector store integrations
uv add llama-index-vector-stores-chroma==0.5.3

# Additional utilities
uv add chromadb sentence-transformers
```

### Environment Configuration

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Verify API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

print("✓ Environment configured")
```

---

## Core Workflow: 5-Phase RAG Pipeline

LlamaIndex automates these five phases:

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Document Loading                                  │
│  ↓ SimpleDirectoryReader / Manual Document creation         │
├─────────────────────────────────────────────────────────────┤
│  Phase 2: Chunking & Embedding                              │
│  ↓ Automatic text splitting + OpenAI embeddings             │
├─────────────────────────────────────────────────────────────┤
│  Phase 3: Index Creation                                    │
│  ↓ VectorStoreIndex / TreeIndex / ListIndex / Keyword       │
├─────────────────────────────────────────────────────────────┤
│  Phase 4: Query Processing                                  │
│  ↓ Embed query → Similarity search → Retrieve context       │
├─────────────────────────────────────────────────────────────┤
│  Phase 5: Response Generation                               │
│  ↓ LLM (GPT-4) generates answer with retrieved context      │
└─────────────────────────────────────────────────────────────┘
```

---

## Document Loading & Preparation

### Method 1: Load from Files (Chapter 3 Pattern)

```python
from llama_index.core import SimpleDirectoryReader

# Load all documents from a directory
documents = SimpleDirectoryReader("./data/").load_data()

print(f"Loaded {len(documents)} documents")
# Output: Loaded 22 documents
```

**What happens:**
- Recursively reads all files in `./data/`
- Supports: `.txt`, `.pdf`, `.md`, `.csv`, etc.
- Each file becomes a `Document` object with:
  - `text`: The content
  - `metadata`: File path, name, size, timestamps
  - `doc_id`: Unique identifier

### Method 2: Manual Document Creation (Chapter 4 Pattern)

```python
from llama_index.core import Document

# Create documents manually (e.g., from database, API, etc.)
documents = []
for data_item in your_data_source:
    doc = Document(
        text=data_item['text'],
        doc_id=data_item['id'],
        metadata={'source': data_item['source']}
    )
    documents.append(doc)
```

### Method 3: Load from Existing Vector Store (Chapter 4 Pattern)

```python
import chromadb
from llama_index.core import Document

# Connect to existing ChromaDB
chroma_client = chromadb.PersistentClient(path="../Chapter03/chroma_db")
chroma_collection = chroma_client.get_collection(name="drone_vision")

# Retrieve documents
results = chroma_collection.get(include=["documents", "metadatas", "embeddings"])

# Convert to LlamaIndex Documents
documents_llm = []
for i in range(len(results['ids'])):
    doc = Document(
        text=results['documents'][i],
        doc_id=results['ids'][i],
        metadata=results['metadatas'][i]
    )
    documents_llm.append(doc)
```

---

## Index Types & When to Use Them

LlamaIndex provides 4 main index types, each optimized for different use cases.

### 1. VectorStoreIndex (Most Common)

**Best for:** Production RAG applications, semantic search

```python
from llama_index.core import VectorStoreIndex

# Create index (automatically embeds documents)
vector_store_index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = vector_store_index.as_query_engine(
    similarity_top_k=3,      # Return top 3 most similar chunks
    temperature=0.1,         # Low temperature for factual responses
    num_output=1024          # Max response length in tokens
)

response = query_engine.query("How do drones identify vehicles?")
print(response)
```

**Performance (Chapter 3 results):**
- Query time: ~5.48s
- Cosine similarity: 0.716
- Performance metric: 0.131

**How it works:**
1. Splits documents into chunks (default: 1024 tokens with 200 overlap)
2. Generates embeddings for each chunk using OpenAI
3. Stores in vector database (in-memory or persistent)
4. Query → Embed → Similarity search → Top-K retrieval → LLM generation

### 2. TreeIndex (Hierarchical Summarization)

**Best for:** Document summarization, hierarchical structures

```python
from llama_index.core import TreeIndex

tree_index = TreeIndex.from_documents(documents)
tree_query_engine = tree_index.as_query_engine(
    similarity_top_k=3,
    temperature=0.1,
    num_output=1024
)

response = tree_query_engine.query("How do drones identify vehicles?")
```

**Performance (Chapter 3 results):**
- Query time: ~5.53s
- Cosine similarity: 0.710
- Performance metric: 0.128

**How it works:**
1. Builds a tree structure from bottom-up
2. Each parent node contains summary of child nodes
3. Query traverses tree from root to relevant leaves
4. Good for finding high-level answers

### 3. ListIndex (Exhaustive Search)

**Best for:** Comprehensive document review, small datasets

```python
from llama_index.core import ListIndex

list_index = ListIndex.from_documents(documents)
list_query_engine = list_index.as_query_engine(
    similarity_top_k=3,
    temperature=0.1,
    num_output=1024
)

response = list_query_engine.query("How do drones identify vehicles?")
```

**Performance (Chapter 3 results):**
- Query time: ~25.64s ⚠️ (Slowest!)
- Cosine similarity: 0.623
- Performance metric: 0.024

**How it works:**
1. Iterates through ALL documents sequentially
2. No indexing optimization
3. Most thorough but slowest
4. Use only for small datasets (<100 documents)

### 4. KeywordTableIndex (Keyword-Based)

**Best for:** Known terminology, keyword search

```python
from llama_index.core import KeywordTableIndex

keyword_index = KeywordTableIndex.from_documents(documents)
keyword_query_engine = keyword_index.as_query_engine(
    similarity_top_k=3,
    temperature=0.1,
    num_output=1024
)

response = keyword_query_engine.query("How do drones identify vehicles?")
```

**Performance (Chapter 3 results):**
- Query time: ~2.36s 🚀 (Fastest!)
- Cosine similarity: 0.762
- Performance metric: 0.322 (Best overall!)

**How it works:**
1. Extracts keywords from each document
2. Creates mapping: keyword → document IDs
3. Query keywords match document keywords
4. Fast but may miss semantic relationships

---

## Query Engines & Retrieval

### Basic Query Pattern

```python
# Step 1: Create index
index = VectorStoreIndex.from_documents(documents)

# Step 2: Create query engine
query_engine = index.as_query_engine(
    similarity_top_k=3,      # Top-K retrieval
    temperature=0.1,         # LLM temperature (0.0 = deterministic, 1.0 = creative)
    num_output=1024          # Max output tokens
)

# Step 3: Query
response = query_engine.query("Your question here")

# Step 4: Access results
print(response)                      # The answer
print(response.source_nodes)         # Retrieved chunks with scores
```

### Accessing Retrieved Context

```python
import time

# Time the query
start_time = time.time()
response = query_engine.query("How do drones identify vehicles?")
elapsed_time = time.time() - start_time

print(f"Query time: {elapsed_time:.4f}s")
print(f"Answer: {response}")

# Examine source nodes
for i, node_with_score in enumerate(response.source_nodes):
    print(f"\nSource {i+1}:")
    print(f"  Score: {node_with_score.score:.4f}")
    print(f"  Node ID: {node_with_score.node.id_}")
    print(f"  Text: {node_with_score.node.text[:200]}...")
```

### Response Object Structure

```python
response = query_engine.query("Your question")

# Available attributes:
response.response          # Final answer string
response.source_nodes      # List[NodeWithScore] - retrieved chunks
response.metadata          # Query metadata
```

---

## Vector Store Integration

LlamaIndex can use external vector databases for persistence and scalability.

### Pattern 1: In-Memory (Default)

```python
# Simplest - no persistence
index = VectorStoreIndex.from_documents(documents)
# ⚠️ Lost when script ends!
```

### Pattern 2: ChromaDB Persistence (Chapter 3 Pattern)

```python
import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

# Configuration
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "drone_vision"

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Create or recreate collection
try:
    chroma_client.delete_collection(name=COLLECTION_NAME)
except:
    pass
chroma_collection = chroma_client.create_collection(name=COLLECTION_NAME)

# Create LlamaIndex wrapper
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index with persistence
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

print(f"✓ Saved to {CHROMA_PATH}")
print(f"  Collection: {COLLECTION_NAME}")
print(f"  Total chunks: {chroma_collection.count()}")
```

### Pattern 3: Load from Existing Vector Store (Chapter 4 Pattern)

```python
# Connect to existing ChromaDB
chroma_client = chromadb.PersistentClient(path="../Chapter03/chroma_db")
chroma_collection = chroma_client.get_collection(name="drone_vision")

# Wrap in LlamaIndex
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load index from existing embeddings (no re-embedding!)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)

print("✓ Loaded existing index without re-embedding")
```

**Comparison:**

| Method | Embedding Required? | Use Case |
|--------|-------------------|----------|
| `from_documents()` | ✅ Yes (always embeds) | New data, first-time indexing |
| `from_vector_store()` | ❌ No (reuses existing) | Load pre-computed embeddings |

---

## Advanced Patterns

### 1. Chunking Configuration

```python
from llama_index.core.node_parser import SentenceSplitter

# Custom chunking strategy
splitter = SentenceSplitter(
    chunk_size=1024,        # Characters per chunk
    chunk_overlap=200       # Overlap between chunks
)

nodes = splitter.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)
```

**Default settings:**
- Chunk size: 1024 tokens (~800 words)
- Overlap: 200 tokens (prevents boundary issues)
- Splits on sentence boundaries (not mid-word)

### 2. Multiple Query Engines (Chapter 4 Pattern)

```python
# LLM text dataset query engine
vector_store_index_llm = VectorStoreIndex.from_documents(documents_llm)
llm_query_engine = vector_store_index_llm.as_query_engine(
    similarity_top_k=2,
    temperature=0.1
)

# Multimodal image dataset query engine
vector_store_index_images = VectorStoreIndex.from_documents(image_documents)
image_query_engine = vector_store_index_images.as_query_engine(
    similarity_top_k=1
)

# Query both
llm_response = llm_query_engine.query("How do drones identify trucks?")
image_response = image_query_engine.query("How do drones identify trucks?")

# Combine results
print(f"Text answer: {llm_response}")
print(f"Image match: {image_response.source_nodes[0].node.text}")
```

### 3. Metadata Filtering

```python
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# Create query engine with filters
query_engine = index.as_query_engine(
    similarity_top_k=3,
    filters=MetadataFilters(filters=[
        ExactMatchFilter(key="file_name", value="article_10.txt")
    ])
)

# Only searches within filtered documents
response = query_engine.query("How do drones work?")
```

### 4. Custom Embedding Models

```python
from llama_index.core import ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding

# Use local embedding model (no OpenAI cost!)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

service_context = ServiceContext.from_defaults(embed_model=embed_model)

index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context
)
```

---

## Best Practices

### 1. Choose the Right Index Type

```python
# Decision tree:
if dataset_size > 10000 or need_production_scale:
    use VectorStoreIndex with ChromaDB/Pinecone
elif need_hierarchical_summaries:
    use TreeIndex
elif dataset_size < 100 and need_comprehensive:
    use ListIndex
elif have_well_defined_keywords:
    use KeywordTableIndex
else:
    use VectorStoreIndex  # Default choice
```

### 2. Optimize Query Parameters

```python
# For factual answers (RAG):
query_engine = index.as_query_engine(
    similarity_top_k=3,      # 3-5 is usually optimal
    temperature=0.1          # Low for factual
)

# For creative generation:
query_engine = index.as_query_engine(
    similarity_top_k=5,      # More context
    temperature=0.7          # Higher for creativity
)
```

### 3. Clean Metadata Before Indexing

```python
# Bad: Large metadata causes chunk size issues
doc = Document(
    text=text,
    metadata=large_metadata_dict  # ⚠️ May cause errors!
)

# Good: Clean unnecessary fields
clean_metadata = {
    k: v for k, v in metadata.items()
    if k not in ['_node_content', '_node_type', 'large_field']
}
doc = Document(text=text, metadata=clean_metadata)
```

**From Chapter 4:**
```python
# Skip large internal fields
clean_metadata = {}
for key, value in results['metadatas'][i].items():
    if key not in ['_node_content', '_node_type']:
        clean_metadata[key] = value
```

### 4. Batch Processing for Large Datasets

```python
# Process documents in batches to avoid memory issues
BATCH_SIZE = 100

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    index = VectorStoreIndex.from_documents(batch)
    # Process or save each batch
```

### 5. Reuse Embeddings When Possible

```python
# ❌ BAD: Re-embeds every time (slow + costly)
for _ in range(10):
    index = VectorStoreIndex.from_documents(documents)

# ✅ GOOD: Embed once, reuse many times
index = VectorStoreIndex.from_documents(documents)
# Save to ChromaDB (see Pattern 2 above)

# Later: Load without re-embedding
index = VectorStoreIndex.from_vector_store(vector_store)
```

---

## Common Patterns Comparison

### Chapter 3: Local RAG with LlamaIndex + ChromaDB

```python
# 1. Load documents from files
documents = SimpleDirectoryReader("./data/").load_data()

# 2. Create persistent index
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.create_collection(name="drone_vision")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# 3. Compare 4 index types
vector_index = VectorStoreIndex.from_documents(documents)
tree_index = TreeIndex.from_documents(documents)
list_index = ListIndex.from_documents(documents)
keyword_index = KeywordTableIndex.from_documents(documents)

# 4. Evaluate performance
for idx in [vector_index, tree_index, list_index, keyword_index]:
    engine = idx.as_query_engine()
    response = engine.query("How do drones identify vehicles?")
    # Measure time, similarity, performance
```

**Key takeaway:** Chapter 3 focuses on **understanding index types** and **local persistence**.

### Chapter 4: Multimodal RAG with Pre-existing Data

```python
# 1. Load pre-embedded data from Chapter 3
chroma_client = chromadb.PersistentClient(path="../Chapter03/chroma_db")
chroma_collection = chroma_client.get_collection(name="drone_vision")
results = chroma_collection.get(include=["documents", "metadatas", "embeddings"])

# 2. Convert to LlamaIndex Documents
documents_llm = []
for i in range(len(results['ids'])):
    doc = Document(
        text=results['documents'][i],
        doc_id=results['ids'][i],
        metadata=clean_metadata(results['metadatas'][i])
    )
    documents_llm.append(doc)

# 3. Create index (will re-embed unless using from_vector_store)
vector_store_index_llm = VectorStoreIndex.from_documents(documents_llm)

# 4. Query for text insights
llm_query_engine = vector_store_index_llm.as_query_engine()
llm_response = llm_query_engine.query("How do drones identify trucks?")

# 5. Combine with image dataset
image_index = VectorStoreIndex.from_documents(image_documents)
image_response = image_index.as_query_engine().query("trucks")

# 6. Multimodal fusion
combined_insight = combine_text_and_image(llm_response, image_response)
```

**Key takeaway:** Chapter 4 focuses on **loading existing embeddings** and **multimodal fusion**.

---

## Performance Metrics (Chapter 3 Benchmark)

Query: "How do drones identify vehicles?"

| Index Type | Query Time | Cosine Similarity | Performance | Rank |
|------------|-----------|-------------------|-------------|------|
| **Keyword Table** | 2.36s 🚀 | 0.762 | 0.322 | **#1** |
| **Vector Store** | 5.48s | 0.716 | 0.131 | #2 |
| **Tree** | 5.53s | 0.710 | 0.128 | #3 |
| **List** | 25.64s ⚠️ | 0.623 | 0.024 | #4 |

**Recommendation:** Use **VectorStoreIndex** for most production applications (balanced speed + quality).

---

## Quick Reference

### Essential Imports

```python
from llama_index.core import (
    VectorStoreIndex,
    TreeIndex,
    ListIndex,
    KeywordTableIndex,
    SimpleDirectoryReader,
    Document,
    StorageContext
)
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
```

### Minimal Working Example

```python
# 1. Load documents
documents = SimpleDirectoryReader("./data/").load_data()

# 2. Create index
index = VectorStoreIndex.from_documents(documents)

# 3. Query
query_engine = index.as_query_engine()
response = query_engine.query("Your question here")

# 4. Get answer
print(response)
```

### Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Metadata length (1113) is longer than chunk size (1024)` | Large metadata fields | Remove `_node_content`, `_node_type` |
| `AttributeError: 'ChatCompletion' object has no attribute 'source_nodes'` | Variable overwriting | Use different variable names |
| `InvalidKeyTypeError: Item '80' of type 'int64' is not a valid key` | Pandas int64 type | Convert with `int(index)` |
| Out of memory during indexing | Too many documents at once | Batch process with `BATCH_SIZE` |

---

## Next Steps

1. **Start with Chapter 3**: Understand index types and local persistence
2. **Move to Chapter 4**: Learn multimodal integration and data reuse
3. **Experiment**: Try different index types on your own data
4. **Optimize**: Measure performance and tune parameters
5. **Scale**: Integrate with production vector databases (Pinecone, Weaviate)

---

## Resources

- **LlamaIndex Documentation**: https://docs.llamaindex.ai/
- **Chapter 3 Notebook**: `Deep_Lake_LlamaIndex_OpenAI_RAG.ipynb`
- **Chapter 4 Notebook**: `Multimodal_Modular_RAG_Drones.ipynb`
- **Q&A**: See `Chapter04/QnA.md` for ChromaDB vs LlamaIndex comparison

---

**Author:** Generated for RAG-Driven Generative AI (Packt Publishing)
**Last Updated:** 2025-11-02
