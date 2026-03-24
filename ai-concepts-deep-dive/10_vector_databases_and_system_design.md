# Vector Databases and System Design

## Where we left off

In [09](./09_vector_embeddings_and_semantic_search.md), we learned how text becomes a meaning vector and how cosine similarity finds semantically related content. But we glossed over a massive practical problem:

**What happens when you have millions or billions of vectors?**

You can't compare the query against every single vector one by one. That would take far too long. This file covers how vector databases solve this problem and how to design a complete semantic search system.

---

## Part 1: The scale problem

### Why brute force doesn't work

Suppose you have 10 million document chunks, each with a 768-dimensional vector. A user submits a query.

Brute force approach:
- Compute cosine similarity between the query vector and all 10 million vectors
- Sort by similarity
- Return the top 10

This requires 10 million similarity computations. Each computation involves multiplying and adding 768 numbers. On a single machine, this might take seconds — which is too slow for a real-time search API that needs to respond in milliseconds.

At 100 million or 1 billion vectors, brute force becomes completely impractical.

### Analogy: finding a book in a library

Brute force is like walking through every shelf in a library, reading every book's summary, and comparing it to what you're looking for. For a small personal bookshelf (100 books), this works fine. For the Library of Congress (170 million items), this is absurd.

What do real libraries do? They organize books into sections, categories, and indices. You go to the relevant section first, then browse locally.

Vector databases do the same thing for vectors.

---

## Part 2: Approximate Nearest Neighbor (ANN) search

### The key insight: trade tiny accuracy for massive speed

Instead of finding the **exact** nearest vectors (which requires checking all of them), ANN algorithms find **approximately** the nearest vectors by only checking a smart subset.

This trade-off is almost always worth it:
- Brute force: checks 10,000,000 vectors, returns perfect top-10, takes 5 seconds
- ANN: checks 10,000 vectors, returns 9 out of 10 correct results, takes 5 milliseconds

For search applications, getting 9/10 correct results in 5ms is far better than 10/10 in 5 seconds.

### How ANN indexes work (conceptual)

There are several ANN algorithms. Let's understand the two most common ones conceptually.

---

### ANN method 1: IVF (Inverted File Index)

#### The idea: cluster first, search within clusters

Before any queries arrive, the system groups all vectors into clusters (say, 1000 clusters) based on proximity.

Think of it like organizing a warehouse into 1000 labeled zones. Zone A has electronics. Zone B has clothing. Zone C has kitchen items. And so on.

```
Offline (building the index):

All 10 million vectors
        |
        v
  Clustering algorithm groups them
  into 1000 clusters based on similarity
        |
        v
  Cluster 1: vectors about technology  (12,000 vectors)
  Cluster 2: vectors about cooking     (8,500 vectors)
  Cluster 3: vectors about sports      (11,200 vectors)
  ...
  Cluster 1000: vectors about finance  (9,800 vectors)

Each cluster has a "center point" (centroid) — the average of all vectors in that cluster.
```

When a query arrives:

```
Online (searching):

Query vector arrives
        |
        v
  Compare query to all 1000 cluster centers
  (only 1000 comparisons — very fast)
        |
        v
  Pick the 5 closest clusters
  (say: clusters about technology, cloud, and DevOps)
        |
        v
  Search only within those 5 clusters
  (maybe 50,000 vectors instead of 10,000,000)
        |
        v
  Return top-10 from those 50,000
```

Instead of searching 10 million vectors, you search ~50,000. That's 200x faster, and the results are almost always correct because the relevant vectors are very likely in the nearest clusters.

#### Analogy: department store

You want running shoes. You don't search every floor. You go to the sports department (nearest cluster) and browse there. You might miss a pair that was accidentally shelved in the fashion department, but you'll find what you need quickly.

---

### ANN method 2: HNSW (Hierarchical Navigable Small World)

#### The idea: build a multi-level graph for efficient navigation

HNSW organizes vectors into a graph structure with multiple levels, like a network of highways, city roads, and local streets.

Think of navigating from New York to a specific restaurant in San Francisco:

- **Top level (highways):** You only see major cities. New York → Chicago → Denver → San Francisco. A few big jumps get you to the right region.
- **Middle level (city roads):** Now in San Francisco, you see neighborhoods. Downtown → Mission District → specific block.
- **Bottom level (local streets):** On the right block, you walk to the specific restaurant.

HNSW works the same way with vectors:

```
Top level:    Few vectors, widely spaced → big jumps
                    |
Middle level: More vectors → medium jumps
                    |
Bottom level: All vectors → fine-grained local search
```

When a query arrives:
1. Start at the top level with a random entry point
2. Jump to the nearest neighbor at that level
3. Drop to the next level and repeat
4. At the bottom level, do a thorough local search

The result: you navigate to the right region of the vector space very quickly (using the top levels) and then find the precise nearest neighbors locally (at the bottom level).

#### Why HNSW is popular

It's fast, accurate, and works well in practice. Many production vector databases (Qdrant, Weaviate, pgvector) use HNSW as their primary index.

---

## Part 3: What is a vector database?

A vector database is a specialized storage system built for:
- **Storing** large numbers of vectors along with their metadata and original text
- **Indexing** those vectors using ANN algorithms for fast search
- **Querying** to find nearest neighbors quickly
- **Filtering** results by metadata (date, category, source, etc.)
- **Updating** vectors as documents are added, changed, or deleted

### Why not just use a regular database?

You could store vectors in PostgreSQL as arrays. But regular databases are designed for exact matching ("give me all rows where country = 'India'") and range queries ("all orders above $100"). They're not designed for "find the 10 closest vectors to this query vector using cosine similarity."

Vector databases add:
- ANN index structures (IVF, HNSW, etc.) that regular databases don't have
- Optimized similarity computation on GPUs
- Specialized data layouts for high-dimensional vectors
- Built-in support for common operations like normalization and distance metrics

### The exception: pgvector

pgvector is a PostgreSQL extension that adds vector indexing to a regular database. It's a middle ground — you get the familiarity of PostgreSQL with basic vector search capabilities. Good for smaller-scale applications. For very large scale (hundreds of millions of vectors), dedicated vector databases are usually faster.

---

## Part 4: Metadata filtering (combining vector search with structure)

Real search needs are rarely "find similar text" alone. They usually include constraints:

- "Find similar support tickets **from the last 30 days**"
- "Find related research papers **in the biology category**"
- "Find matching job descriptions **for senior roles in engineering**"

This is where metadata filtering comes in.

### How it works

Each vector is stored with metadata:

```
Vector: [0.45, -0.22, 0.67, ...]
Metadata: {
  source: "support_tickets",
  date: "2026-03-15",
  category: "billing",
  priority: "high"
}
Original text: "Customer reports double charge on monthly invoice"
```

When searching, you can combine:
- **Vector similarity:** find semantically similar content
- **Metadata filter:** restrict to specific categories, dates, or other attributes

Example query: "billing overcharge issues" + filter: `date > 2026-01-01 AND priority = "high"`

The vector database first applies the filter (narrowing candidates), then runs similarity search on the filtered subset (or does both simultaneously, depending on the implementation).

---

## Part 5: Hybrid search (combining keyword + semantic)

Pure semantic search isn't always best. Pure keyword search isn't always best. Many production systems combine both.

### When keyword search is better

- Searching for specific error codes: "ERR-4521"
- Searching for exact product names: "MacBook Pro M3"
- Searching for specific identifiers: "JIRA-12345"

Semantic search might match similar concepts but miss the exact string.

### When semantic search is better

- "How to fix slow API responses" (concept-based, many phrasings)
- "What's our policy on remote work" (intent-based, not keyword-dependent)

### Hybrid approach

Run both keyword search and semantic search in parallel, then combine results.

```
User query: "fix ERR-4521 in payment service"

Keyword search results:
  1. Log entry mentioning "ERR-4521"
  2. Ticket about "ERR-4521 resolution"

Semantic search results:
  1. Article about "payment service error handling"
  2. Guide on "debugging transaction failures"

Combined results (de-duplicated, re-ranked):
  1. Ticket about "ERR-4521 resolution"  (matched by both)
  2. Log entry mentioning "ERR-4521"
  3. Article about "payment service error handling"
  4. Guide on "debugging transaction failures"
```

The combined result is better than either alone.

---

## Part 6: Re-ranking (optional but powerful)

After initial retrieval (whether keyword, semantic, or hybrid), a **re-ranker** can improve result quality.

### Why re-ranking helps

Initial retrieval is fast but approximate. The embedding model compresses an entire text into one vector — some nuance is lost. A re-ranker takes the query and each candidate result, reads them together more carefully, and re-scores relevance.

### Analogy: job hiring

- **Initial retrieval = resume screening:** Quick scan to find 50 promising candidates from 10,000 applications. Fast but imperfect.
- **Re-ranking = interviews:** Carefully evaluate the top 50 candidates. Slower but much more accurate.

You can't interview 10,000 people (too slow). You can't hire from resumes alone (too shallow). The two-stage approach gives you both speed and quality.

### How re-rankers work

A re-ranker is typically a model that takes **both the query and a candidate document** as input and produces a single relevance score.

Unlike the embedding model (which encodes query and document separately), the re-ranker sees them **together**, allowing it to detect fine-grained relevance that separate embeddings might miss.

---

## Part 7: Designing a complete semantic search system

Now let's put everything together into a system design.

### The 7 questions to answer

1. **What data?** — Documents, tickets, wikis, code, knowledge base articles?
2. **How to chunk?** — Chunk size (300-800 tokens), overlap (10-20%)
3. **Which embedding model?** — General purpose vs domain-specific
4. **Which vector database?** — Based on scale, latency, budget
5. **What metadata?** — Source, date, category, author, etc.
6. **Hybrid search?** — Do you need keyword + semantic?
7. **Evaluation?** — How will you measure quality?

### Complete system architecture

```
Data Sources (docs, tickets, wikis)
        |
        v
  Data Pipeline
  (extract, clean, chunk)
        |
        v
  Embedding Model
  (chunk text → vectors)
        |
        v
  Vector Database
  (store vectors + metadata + text)
        |
        |
        |  ← User Query arrives
        |
        v
  Query Pipeline:
    1. Embed the query (same model)
    2. Vector search (ANN nearest neighbors)
    3. Metadata filter (if needed)
    4. (Optional) Keyword search in parallel
    5. (Optional) Re-rank top candidates
    6. Return results
        |
        v
  (Optional) LLM generates answer using retrieved chunks (RAG)
```

### What is RAG?

RAG = Retrieval-Augmented Generation. It combines semantic search with a language model:

1. User asks a question
2. System retrieves relevant document chunks via semantic search
3. Retrieved chunks are given to an LLM as context
4. LLM generates an answer based on the retrieved information

This gives the LLM access to your specific data (which it wasn't trained on) and reduces hallucination because the answer is grounded in real documents.

### Analogy: open-book exam

Without RAG, the LLM is taking a closed-book exam — it can only use what it memorized during training. With RAG, it's an open-book exam — the model can look up relevant pages before answering.

---

## Part 8: Common pitfalls and how to avoid them

### Pitfall 1: Bad chunking
If chunks are too small, they lose context. If too large, they contain mixed topics and the embedding becomes diluted.

**Fix:** Experiment with chunk sizes. Use semantic boundaries (paragraphs, sections) when possible rather than arbitrary character counts.

### Pitfall 2: Wrong embedding model
A model trained on academic papers may not work well for support tickets. Domain mismatch hurts retrieval quality.

**Fix:** Test your embedding model on your actual data. Measure retrieval quality with real queries.

### Pitfall 3: No evaluation
Without measuring quality, you're flying blind. You might think search is "good enough" when it's actually missing important results.

**Fix:** Create a test set of queries with known relevant documents. Measure precision (how many results are relevant) and recall (how many relevant documents were found).

### Pitfall 4: Ignoring metadata filters
Without filters, a search for "billing issues" might return results from every department, time period, and priority level.

**Fix:** Store useful metadata. Let users or the system filter by relevant dimensions.

### Pitfall 5: Treating ANN as exact
ANN is approximate. Tuning index parameters (number of clusters, graph connectivity) affects the accuracy-speed tradeoff.

**Fix:** Understand your vector database's index settings. Test with known queries to verify recall.

---

## Part 9: Choosing a vector database

| Database | Type | Best for |
|----------|------|----------|
| Pinecone | Managed cloud service | Quick start, no infrastructure management |
| Qdrant | Open source / cloud | High performance, rich filtering |
| Weaviate | Open source / cloud | Hybrid search, integrated ML pipeline |
| Milvus | Open source | Very large scale (billions of vectors) |
| pgvector | PostgreSQL extension | Smaller scale, existing PostgreSQL stack |
| Elasticsearch | Search engine with vector support | Existing Elastic stack, hybrid keyword + vector |
| OpenSearch | AWS-managed search with vector support | AWS ecosystem, hybrid search |

The "best" choice depends on scale, existing infrastructure, budget, and team expertise.

---

## Summary: the full chain from text to search result

```
"How to reduce cloud storage costs"
        |
        v
  Tokenizer: ["How", "to", "reduce", "cloud", "storage", "costs"]
        |
        v
  Embedding table: each token gets initial 768-number vector
        |
        v
  Position encoding: order information added
        |
        v
  Transformer layers (attention + MLP × N):
    - Tokens gather context from each other
    - Each token refines its representation
    - Understanding deepens layer by layer
        |
        v
  Pooling: average all token vectors into one 768-number vector
        |
        v
  Normalized vector: [0.41, -0.25, 0.69, ...]
        |
        v
  ANN search in vector database: find closest vectors
        |
        v
  Results: documents with similar meaning, ranked by cosine similarity
        |
        v
  (Optional) Re-rank for better precision
        |
        v
  (Optional) Feed to LLM for answer generation (RAG)
```

That's the complete journey from a human question to a meaningful search result — and every step along the way connects back to the concepts in files 01 through 09.

---

**Previous: [09 — Vector Embeddings and Semantic Search](./09_vector_embeddings_and_semantic_search.md)**  
**Start over: [01 — Tokens and Embeddings](./01_tokens_and_embeddings.md)**
