# Vector Embeddings and Semantic Search

## Where we left off

We've built up a complete understanding:
- How text becomes numbers ([01](./01_tokens_and_embeddings.md))
- How tokens gather context from each other ([02](./02_attention.md))
- How tokens refine their understanding ([03](./03_mlp_feed_forward.md))
- How depth stays stable ([04](./04_residual_connections_and_layer_norm.md))
- How the full system assembles ([05](./05_transformer_block_and_layers.md))
- How older models worked and why transformers replaced them ([06](./06_rnn_and_lstm.md))
- The different transformer architectures ([07](./07_encoder_vs_decoder.md))
- How the model learns all this from data ([08](./08_training_and_learning.md))

Now we connect everything to the practical application: **using transformer outputs to search by meaning instead of keywords.**

---

## The problem: keyword search fails at meaning

Traditional search engines work by matching exact words.

You search for: **"how to reduce cloud storage costs"**

Keyword search finds documents containing those words. But it **misses** a great article titled:  
**"Cutting your S3 bills with lifecycle policies"**

Why? Because none of the keywords match. "Reduce" vs "cutting." "Cloud storage" vs "S3." "Costs" vs "bills."

A human reads both and immediately knows they're about the same topic. But keyword search can't see that.

**Semantic search** solves this. It searches by meaning, not by exact words.

---

## How semantic search works (the big picture)

The core idea is simple:

1. Convert text into a vector (list of numbers) that captures its meaning
2. Convert the search query into a vector the same way
3. Find which text vectors are closest to the query vector
4. Return those texts as results

"Closest" means "most similar in meaning." Two texts about the same topic will have similar vectors, even if they use completely different words.

---

## Step 1: How text becomes a meaning vector

This is where everything we've learned comes together.

### The embedding pipeline

```
Input text: "How to reduce cloud storage costs"
                    |
                    v
              Tokenizer
  (splits into subword tokens)
                    |
                    v
        Embedding + Position
  (each token gets initial vector)
                    |
                    v
         Transformer Layers
  (attention gathers context,
   MLP refines understanding,
   repeated many times)
                    |
                    v
      Token-level representations
  (each token now has a rich,
   context-aware vector)
                    |
                    v
              Pooling
  (combine all token vectors
   into ONE sentence vector)
                    |
                    v
     Sentence embedding: [0.12, -0.87, 0.43, ...]
```

Everything from files 01 through 05 is in that pipeline. The new step is **pooling** — combining many token vectors into one.

### What is pooling?

After the final transformer layer, you have one vector per token. If your sentence has 8 tokens and each vector has 768 numbers, you have 8 vectors of 768 numbers each.

But for search, you need **one vector for the entire sentence.** You need to collapse those 8 vectors into 1.

The most common method is **mean pooling** — simply averaging all token vectors.

### Analogy: averaging student scores

Imagine 8 students each took a test with 768 questions. Each student has a score for each question. To get one "class profile," you average each question's scores across all students.

The result is one profile (768 numbers) that represents the entire class, not any individual student.

Mean pooling does this for token vectors. The result is one vector that represents the meaning of the entire sentence.

### Why does averaging work?

Because each token's vector has been enriched by attention through multiple layers. By the final layer, each token's vector already contains information about the full sentence (because attention let it gather context from all other tokens). Averaging these context-rich vectors produces a good summary of the overall meaning.

### Other pooling methods

- **CLS token pooling:** Some models add a special token at the beginning (called [CLS]). This token attends to all others, so its final vector is treated as the sentence summary. Used by BERT.
- **Max pooling:** Instead of averaging, take the maximum value across tokens for each dimension. Less common but sometimes useful.
- **Learned pooling:** A small additional layer that learns how to best combine token vectors. Used in some advanced models.

Mean pooling is the most common in modern sentence embedding models because it works well and is simple.

---

## Step 2: Measuring similarity between vectors

Once you have vectors for your documents and your query, you need to measure which documents are "closest" in meaning.

### Cosine similarity (the most common measure)

Cosine similarity measures the **angle** between two vectors, ignoring their length.

### Analogy: compass directions

Imagine two people standing at the center of a field, each pointing in a direction.

- If they point in the **same direction** — they agree completely. Cosine similarity = 1.
- If they point in **opposite directions** — they completely disagree. Cosine similarity = -1.
- If they point at **right angles** — they're unrelated. Cosine similarity = 0.

The length of their arm doesn't matter — only the direction they point.

For text vectors:
- "How to reduce cloud costs" and "Ways to lower S3 bills" point in nearly the same direction → high cosine similarity (close to 1)
- "How to reduce cloud costs" and "Best pizza in New York" point in very different directions → low cosine similarity (close to 0)

### Why direction matters more than magnitude

Two documents about the same topic should be considered similar regardless of their length or other surface differences. Cosine similarity captures this by focusing on the direction (what the content is about) rather than the magnitude (how long or detailed it is).

### Other similarity measures

- **Dot product:** Similar to cosine but also considers vector magnitude. Useful when you want longer/more-confident representations to rank higher.
- **Euclidean distance (L2):** Geometric straight-line distance between vector endpoints. Smaller distance = more similar.

In practice, most semantic search systems normalize vectors (make them all the same length) and use cosine similarity or dot product (which become equivalent after normalization).

---

## Step 3: The search algorithm

### Offline phase (preparing your data)

Before any search happens, you process all your documents:

```
Document 1: "Cutting your S3 bills with lifecycle policies"
  → Tokenize → Embed → Transformer layers → Pool
  → Vector: [0.45, -0.22, 0.67, ...]

Document 2: "Introduction to Kubernetes networking"
  → Tokenize → Embed → Transformer layers → Pool
  → Vector: [-0.31, 0.88, -0.12, ...]

Document 3: "AWS cost optimization best practices"
  → Tokenize → Embed → Transformer layers → Pool
  → Vector: [0.43, -0.19, 0.71, ...]

... do this for every document in your collection
```

Store all these vectors (plus the original text and any metadata) in a vector database.

### Online phase (answering a search query)

When a user searches:

```
Query: "how to reduce cloud storage costs"
  → Tokenize → Embed → Transformer layers → Pool
  → Query vector: [0.41, -0.25, 0.69, ...]
```

Now compare the query vector against all stored document vectors:

```
Similarity(query, doc1) = 0.94  ← very similar!
Similarity(query, doc2) = 0.12  ← not related
Similarity(query, doc3) = 0.91  ← very similar!
```

Return documents ranked by similarity:
1. Document 1 (0.94) — "Cutting your S3 bills with lifecycle policies"
2. Document 3 (0.91) — "AWS cost optimization best practices"
3. Document 2 (0.12) — "Introduction to Kubernetes networking"

Notice: Document 1 shares **zero keywords** with the query, yet it ranked highest. The model understood that "cutting S3 bills" and "reduce cloud storage costs" are about the same thing. That's the power of semantic search.

---

## The critical requirement: same model for documents and queries

One thing that must be absolutely clear: the **same embedding model** must be used for both documents and queries.

Why? Because each model creates its own "meaning space." Model A might represent "cloud storage" as `[0.41, -0.25, 0.69]`. Model B might represent it as `[-0.82, 0.14, 0.53]`. These are completely incompatible.

It's like GPS coordinates. If your map uses WGS84 coordinates and your navigation uses a different coordinate system, the positions won't match even though they refer to the same locations.

Same model = same coordinate system = meaningful comparisons.

---

## Chunking: handling long documents

Most embedding models have a maximum input length — typically 256 to 512 tokens (some newer models handle 8,192 or more). A long document might be thousands of tokens.

Solution: **chunk** the document into smaller pieces and embed each chunk separately.

### How to chunk

A 10-page document might be split into:
- Chunk 1: first 2 paragraphs
- Chunk 2: next 2 paragraphs (with some overlap with chunk 1)
- Chunk 3: next 2 paragraphs (with some overlap with chunk 2)
- ...

### Why overlap?

Without overlap, important information at chunk boundaries gets split awkwardly. A key sentence might start at the end of chunk 1 and finish at the start of chunk 2. Neither chunk captures the full meaning.

Overlap (typically 10-20% of chunk size) ensures boundary content appears in at least one complete chunk.

### Analogy: photographing a mural

You can't photograph a large mural in one shot. So you take overlapping photos of sections. Each photo captures its section completely, and the overlap ensures no part of the mural falls between photos.

Chunking is the same — overlapping sections ensure complete coverage of meaning.

---

## How [0.12, -0.87, 0.43, ...] was generated (the full story)

Now we can finally give the complete answer to the original question.

When an embedding model produces a vector like `[0.12, -0.87, 0.43, ...]`, here's everything that happened:

1. **Tokenizer** split the text into subword pieces based on a learned vocabulary.

2. **Embedding table** looked up each token ID and retrieved a 768-number vector that was learned during training to capture basic identity.

3. **Position encoding** added order information so the model knows sequence.

4. **Layer 1 attention** let each token look at all others and gather relevant context using learned question/label/content cards. Residual connections preserved initial info. Layer norm kept numbers stable.

5. **Layer 1 MLP** refined each token's representation using the gathered context — creating new features through nonlinear transformation.

6. **Layers 2 through 12** (or however many) repeated this process, each time deepening understanding: grammar → phrases → semantics → full meaning.

7. **Pooling** averaged all final token vectors into one sentence-level vector.

8. **Normalization** scaled the vector to unit length for consistent similarity comparison.

9. The resulting `[0.12, -0.87, 0.43, ...]` is a coordinate in a high-dimensional meaning space where proximity represents semantic similarity.

10. These specific values exist because the model was trained on billions of text examples using contrastive learning — rewarding vectors that are close for similar texts and far for different texts.

Every number in that vector is the cumulative result of billions of parameter adjustments during training. No number was hand-designed. No dimension has a clean human-interpretable meaning. But the pattern across all dimensions collectively encodes the meaning of the input text.

---

## Summary

| Step | What happens | Analogy |
|------|-------------|---------|
| Tokenize | Text split into subwords | Breaking a sentence into puzzle pieces |
| Embed + Position | Tokens get initial vectors with order info | Giving each piece an identity badge and seat number |
| Transformer layers | Attention gathers context; MLP refines understanding | Multiple rounds of group discussion + private reflection |
| Pooling | Many token vectors averaged into one sentence vector | Averaging all students' scores into one class profile |
| Normalization | Vector scaled to consistent length | Standardizing to a common measurement unit |
| Similarity comparison | Cosine of angle between two vectors | How closely two people point in the same direction |
| Semantic search | Find nearest vectors to the query vector | Finding the closest books on a meaning-organized shelf |
| Chunking | Long documents split into overlapping pieces | Overlapping photos of a large mural |

---

**Previous: [08 — Training and Learning](./08_training_and_learning.md)**  
**Next: [10 — Vector Databases and System Design](./10_vector_databases_and_system_design.md)** — storing, indexing, and searching vectors at scale in production.
