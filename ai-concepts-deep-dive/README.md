# AI Concepts Deep Dive

A 10-part beginner-friendly series that teaches the core concepts behind transformers, vector embeddings, and semantic search — from absolute zero to system design.

No formula-first approach. Every concept is explained through analogies, real examples, and logical buildup before any technical terms are introduced. Each file builds on the previous one, so read them in order.

---

## Reading Order

### [01 — Tokens and Embeddings](./01_tokens_and_embeddings.md)

**The starting point: how does a computer read text?**

Computers don't understand words — they only understand numbers. This section explains how raw text is broken into tokens (subword pieces), how each token is converted into a vector of learned numbers (an embedding), and why position information must be added so the model knows word order. You'll understand why "unbelievable" is split into "un" + "believ" + "able" and what the embedding table actually looks like.

---

### [02 — Attention](./02_attention.md)

**The core innovation: how each word decides which other words matter.**

After tokens get their initial vectors, they still don't know about context. "Bank" has the same vector whether it's "river bank" or "bank approved the loan." Attention fixes this. Using a detective-at-a-crime-scene analogy, this section walks through how every token creates a question card (Query), a label card (Key), and a content card (Value), how relevance scores are computed, and how each token builds a context-aware representation by gathering weighted information from all other tokens. Multiple attention heads are explained as a team of analysts with different specialties.

---

### [03 — MLP / Feed-Forward Block](./03_mlp_feed_forward.md)

**What happens after gathering context: the thinking step.**

Attention collects ingredients; MLP cooks. This section explains why gathering information (attention) isn't enough — the model also needs to process and transform that information. The MLP's three-step process (expand, activate, compress) is explained through microscope, highlighting, and summarization analogies. You'll understand why this nonlinear transformation is essential and where factual knowledge actually lives inside the model.

---

### [04 — Residual Connections and Layer Normalization](./04_residual_connections_and_layer_norm.md)

**The hidden infrastructure that makes deep networks possible.**

Stacking many layers can break things — numbers can explode, vanish, or lose earlier information. This section explains residual connections (track changes on a document, not full rewrites) and layer normalization (standardizing test scores across teachers). Neither is glamorous, but without them, training a 96-layer transformer would be impossible. The highway-with-speed-limits analogy ties both concepts together.

---

### [05 — Transformer Block and Layers](./05_transformer_block_and_layers.md)

**Putting all the pieces together into one working system.**

Now that you know attention, MLP, residuals, and layer norm individually, this section assembles them into a complete transformer block and explains what happens when you stack many blocks. Using a book-club-discussion analogy (each meeting = one layer) and a portrait-painting analogy (each pass adds depth), you'll understand how early layers capture grammar, middle layers build phrase-level meaning, and late layers achieve full semantic understanding. Includes the complete system diagram from raw text to final output.

---

### [06 — RNN and LSTM](./06_rnn_and_lstm.md)

**The older approaches: understanding what came before transformers and why they were replaced.**

RNNs read text one word at a time, carrying a rolling memory — like listening to a story on the phone. LSTMs improved on this with gated memory (forget gate, input gate, output gate) — like having better note-taking rules. This section traces our example sentence through both architectures, explains the vanishing gradient problem through a telephone-game analogy, and shows concretely why transformers' direct attention is superior to sequential memory compression.

---

### [07 — Encoder vs Decoder](./07_encoder_vs_decoder.md)

**Three ways to build a transformer, each suited for different tasks.**

Should each token see the full sentence, or only what came before it? This single design choice creates encoder-only (BERT — for understanding), decoder-only (GPT — for generation), and encoder-decoder (T5 — for translation/summarization) variants. Explained through exam-vs-creative-writing and conference-interpreter analogies. Includes a task-to-architecture mapping table and why encoder-based models are the standard choice for producing search embeddings.

---

### [08 — Training and Learning](./08_training_and_learning.md)

**How the model learns everything it knows — from random numbers to language understanding.**

Every number in the model starts random. Through billions of predict-check-adjust cycles, patterns emerge. This section traces one full training step: forward pass (student takes a quiz), loss calculation (grading), backward pass (tracing a factory defect to its source), and weight update (tuning a guitar string). Covers next-token prediction for decoders, masked language modeling for encoders, and contrastive learning (organizing a library by topic) for embedding models. Explains where knowledge lives and why models can confidently say wrong things.

---

### [09 — Vector Embeddings and Semantic Search](./09_vector_embeddings_and_semantic_search.md)

**Connecting transformer representations to real search systems.**

This is where everything comes together. The full pipeline: text → tokens → embeddings → transformer layers → pooling → one sentence vector. Explains mean pooling (averaging student scores into a class profile), cosine similarity (compass directions), chunking long documents (overlapping photos of a mural), and walks through a complete search example where "how to reduce cloud storage costs" matches "Cutting your S3 bills with lifecycle policies" despite sharing zero keywords. Includes the definitive answer to "how were the numbers `[0.12, -0.87, 0.43, ...]` generated?"

---

### [10 — Vector Databases and System Design](./10_vector_databases_and_system_design.md)

**Storing, indexing, and searching vectors at scale in production.**

Brute-force comparison against millions of vectors is too slow. This section explains Approximate Nearest Neighbor (ANN) search through two major algorithms: IVF (department store — go to the right section first) and HNSW (highway navigation — big jumps then local search). Covers metadata filtering, hybrid search (keyword + semantic), re-ranking (resume screening then interviews), RAG (open-book exam for LLMs), common pitfalls, and a complete production system architecture diagram. Includes a vector database comparison table for choosing the right tool.

---

## Who is this for?

- Engineers who want to understand AI/ML concepts without drowning in math
- Anyone building or evaluating semantic search, RAG, or embedding-based systems
- People who've read explanations full of jargon and formulas and still didn't "get it"

## How to read this

Start at 01 and go in order. Each file defines every term before using it, builds on the previous file, and uses consistent analogies throughout. Skip nothing — each concept is a building block for what follows.
