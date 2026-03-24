# Transformer Block and Layers

## Where we left off

We've learned four components individually:
- [01 — Tokens and embeddings](./01_tokens_and_embeddings.md): how text becomes numbers
- [02 — Attention](./02_attention.md): how tokens gather context from each other
- [03 — MLP](./03_mlp_feed_forward.md): how each token refines its understanding
- [04 — Residuals and layer norm](./04_residual_connections_and_layer_norm.md): how depth stays stable

Now it's time to **assemble** these parts into one working system and understand why stacking layers creates increasingly deep understanding.

---

## What is a transformer block?

A transformer block (also called a layer) is one complete cycle of:

1. Attention (gather context)
2. MLP (refine understanding)

With residual connections and layer normalization supporting both.

That's it. One block = one round of "gather then refine."

---

## The flow inside one block

Let's trace what happens to our sentence's tokens through one block.

Starting point: each token has a vector (from previous layer, or from embedding if this is the first layer).

```
Token vectors come in
        |
        v
   Layer Norm
        |
        v
   Attention
   (each token gathers context from all others)
        |
        v
   Add back original (residual connection)
        |
        v
   Layer Norm
        |
        v
   MLP
   (each token refines its own representation)
        |
        v
   Add back previous (residual connection)
        |
        v
Token vectors go out (richer than when they came in)
```

Every token enters the block with some level of understanding. Every token exits the block with deeper understanding.

---

## Analogy: one round of a book club discussion

Imagine a book club where 10 people are reading the same novel. Each person has their own interpretation of the story so far.

**One block = one meeting:**

1. **Attention phase (the discussion):** Each person listens to everyone else's perspective. Some perspectives are more relevant to their own questions than others. Each person mentally weights what they hear — paying close attention to the insightful comments, less attention to off-topic remarks. By the end of the discussion, each person has absorbed relevant perspectives from others.

2. **MLP phase (personal reflection):** After the discussion, each person goes home and thinks privately. They combine what they already knew with the new perspectives they gathered. They might notice connections they hadn't seen before. They write updated notes.

3. **Residual aspect:** Importantly, they don't throw away their original notes. They add the new insights on top of what they already had.

After one meeting, everyone has a richer understanding than before. But one meeting isn't enough for deep understanding.

---

## Why multiple layers (stacking blocks)

A single block does one round of gather + refine. That helps, but complex language needs multiple rounds.

### Layer-by-layer understanding (our running example)

Sentence: **"The animal didn't cross the street because it was too tired."**

**After layer 1-2 (early layers):**
- Basic relationships form
- "didn't" connects to "cross" (negation + verb)
- "the" connects to "animal" and "street" (articles + nouns)
- "too" connects to "tired" (degree modifier)
- "it" starts weakly connecting to both "animal" and "street"

Think of this like first read of a paragraph — you get the basic structure.

**After layer 3-6 (middle layers):**
- Phrase-level meaning emerges
- "didn't cross the street" understood as a complete action phrase
- "too tired" understood as a reason/state
- "it" starts preferring "animal" over "street" because "tired" aligns with animate things
- Causal structure ("because") links the two halves

Think of this like re-reading and starting to see the deeper connections.

**After layer 7-12 (later layers):**
- Full semantic understanding
- "it" is strongly resolved to "animal"
- The full meaning is clear: the animal failed to cross the street due to exhaustion
- Representation is now ready for the final task (prediction, classification, or embedding)

Think of this like studying the text carefully and having full comprehension.

### Why can't one layer do all of this?

Because understanding builds hierarchically. You can't understand the full causal structure of a sentence without first understanding the basic grammatical relationships. And you can't resolve pronoun references without first understanding phrase boundaries.

Each layer builds on what the previous layer established. Early layers handle simple patterns. Later layers handle complex ones. This is not hard-coded — it's a pattern that emerges from training.

### Analogy: painting a portrait

A painter doesn't create a finished portrait in one stroke.

- **First pass:** rough sketch of proportions (basic structure)
- **Second pass:** add major features — eyes, nose, mouth placement (phrase-level understanding)
- **Third pass:** add shading, depth, expression (semantic nuance)
- **Fourth pass:** fine details — individual hairs, skin texture, light reflection (task-ready representation)

Each pass builds on the previous one. Skipping passes creates an incomplete picture. More passes (within reason) create richer output.

Transformer layers work the same way. Each layer is one pass of refinement over the entire sequence.

---

## How many layers do real models have?

This varies by model size and purpose:

- Small models: 6-12 layers
- Medium models: 24-32 layers
- Large models (GPT-4 class): 80-120+ layers

More layers = more rounds of refinement = potentially deeper understanding. But also more computation, more memory, slower inference.

There's a practical tradeoff: at some point, adding more layers helps less and less. The first 12 layers might capture 80% of the understanding. Layers 13-48 refine the remaining 20%. The model designer chooses based on quality needs, cost, and speed.

---

## The full transformer architecture (zooming out)

Now let's see the entire system from raw text to final output:

```
"The animal didn't cross the street because it was too tired."
                          |
                          v
                    Tokenizer
        (text -> token IDs: [464, 5765, ...])
                          |
                          v
              Embedding + Position
        (IDs -> initial vectors with order info)
                          |
                          v
              ┌─────────────────────┐
              │   Transformer       │
              │   Block 1           │
              │   (Attention + MLP) │
              └─────────┬───────────┘
                        |
              ┌─────────────────────┐
              │   Transformer       │
              │   Block 2           │
              │   (Attention + MLP) │
              └─────────┬───────────┘
                        |
                       ...
                        |
              ┌─────────────────────┐
              │   Transformer       │
              │   Block N           │
              │   (Attention + MLP) │
              └─────────┬───────────┘
                        |
                        v
                  Output Head
        (final vectors -> task result)
```

The output head depends on the task:
- For text generation: predicts next token
- For classification: predicts a label
- For embeddings: produces a sentence vector

We'll cover these different uses in [07 — Encoder vs Decoder](./07_encoder_vs_decoder.md).

---

## Important insight: all blocks share the same structure but different learned patterns

Every block has the same architecture: attention + MLP with residuals and layer norm. But the **learned weights** (numbers) inside each block are different.

Think of it like multiple branches of the same restaurant chain. Same kitchen layout, same equipment — but different chefs. Each chef has their own style and specialty. Early-layer chefs handle basic prep. Later-layer chefs handle complex plating and flavor balancing.

The model learns during training which patterns each layer should specialize in. Nobody programs this — it emerges from the training process.

---

## Common question: does information flow only forward?

During **inference** (using the model), yes — information flows forward through layers, one after another.

During **training**, error signals flow **backward** through all layers (backpropagation). This is how each layer learns what patterns to detect. Residual connections are crucial here — they give error signals a clean path to travel backward without degrading.

---

## Putting it all together: one sentence, full journey

Let's trace **"The animal didn't cross the street because it was too tired."** through the entire system one more time, with everything we've learned:

**1. Tokenization:**  
Text becomes token IDs. Each piece of text has a number the model can look up.

**2. Embedding + Position:**  
Each token ID gets a vector from the embedding table. Position information is added. At this point, each token knows "what it is" and "where it sits" but nothing about context.

**3. Layer 1 (Attention):**  
Every token looks at every other token using question/label/content cards. Basic connections form: articles attach to nouns, "didn't" links to "cross."

**3b. Layer 1 (MLP):**  
Each token privately refines what it gathered. The gathered context is processed into more useful features. Residual connections preserve earlier info. Layer norm keeps numbers stable.

**4. Layer 2-4 (repeating):**  
Each layer repeats the gather-and-refine cycle. Understanding deepens each time. Phrase boundaries become clear. "too tired" is understood as a unit. "it" begins to resolve.

**5. Layer 5-12 (deeper layers):**  
Pronoun resolution solidifies. Causal structure is captured. The representation of each token now encodes its full role in the sentence.

**6. Output:**  
Final token representations are used for whatever the task is — predicting the next word, classifying sentiment, or generating an embedding vector.

---

## Summary

| Concept | What it is | Analogy |
|---------|-----------|---------|
| Transformer block | One cycle of attention + MLP (with residuals and norm) | One book club meeting |
| Multiple layers | Repeated blocks that deepen understanding | Multiple editing passes on a painting |
| Early layers | Capture basic grammar and local patterns | Sketching rough proportions |
| Middle layers | Build phrase-level and relational meaning | Adding major features and shading |
| Late layers | Full semantic understanding, task-ready features | Fine details and finishing touches |
| Full system | Tokenize -> Embed -> Stack blocks -> Output | Raw material -> Refinery stages -> Final product |

---

**Previous: [04 — Residual Connections and Layer Normalization](./04_residual_connections_and_layer_norm.md)**  
**Next: [06 — RNN and LSTM](./06_rnn_and_lstm.md)** — understanding the older approaches and why transformers replaced them.
