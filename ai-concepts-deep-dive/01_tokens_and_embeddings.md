# Tokens and Embeddings

## The problem: computers don't understand text

A computer has no idea what "hello" means. It doesn't read English. It doesn't understand Hindi. It doesn't understand any language at all. It only understands numbers.

So the very first challenge in building any AI language model is:

> How do we convert human text into numbers that a computer can work with — in a way that preserves meaning?

This is solved in two stages: **tokenization** and **embedding**.

---

## Stage 1: Tokenization

### What it does

Tokenization breaks raw text into small pieces called **tokens**.

### Why not just use whole words?

You might think: just assign a number to every word. "cat" = 1, "dog" = 2, "hello" = 3, and so on.

The problem is the real world has:

- Millions of words across languages
- New words constantly ("ChatGPT" didn't exist in 2019)
- Typos, slang, technical terms, code
- Variations: "running", "runs", "runner", "ran"

If you tried to have one number per word, your vocabulary would be impossibly large, and any new word would be completely unknown to the model.

### The solution: subword tokenization

Instead of full words, modern models break text into **subword pieces**. These are fragments that the model has learned are useful building blocks.

Example:

The word "unbelievable" might be split into:

```
"un" + "believ" + "able"
```

Why is this clever?

- The model knows "un" as a prefix (means "not")
- The model knows "believ" as a root (related to "believe")
- The model knows "able" as a suffix (means "capable of")

So even if the model has never seen the exact word "unbelievable" before, it can understand it from its parts — just like you can.

Another example — a word the model has definitely never seen in training:

"Transformerification" might become:

```
"Trans" + "former" + "ific" + "ation"
```

The model can still work with it because it knows the pieces.

### What a tokenizer actually produces

The tokenizer has a fixed vocabulary — let's say 50,000 subword pieces. Each piece has an ID number.

So the sentence:

```
"The animal didn't cross the street"
```

Might become token IDs:

```
[464, 5765, 1422, 3272, 464, 4891]
```

These IDs are just lookup numbers. They carry **no meaning yet**. They're like student roll numbers — useful for identification, but they don't tell you anything about the student.

### Key takeaway

Tokenization converts text into a sequence of ID numbers. These IDs let the model reference specific text pieces, but the IDs themselves are meaningless.

---

## Stage 2: Embeddings

### The problem tokenization doesn't solve

After tokenization, we have numbers like `[464, 5765, 1422, ...]`. But these are just arbitrary IDs.

- Token 464 and token 465 are not "close" in meaning just because their IDs are close
- Token 5765 being a bigger number than 464 means nothing
- You can't do math on these IDs and get anything useful

We need a way to represent each token so that the numbers actually capture something about meaning.

### What an embedding is

An embedding is a **list of numbers** (a vector) assigned to each token, where the numbers are **learned during training** to capture meaningful properties.

Instead of one ID number per token, we have many numbers per token — typically 768, 1024, or even 4096 numbers.

Example (simplified to 4 numbers for illustration):

```
"cat"     -> [0.8, -0.1, 0.3, 0.5]
"dog"     -> [0.7, -0.2, 0.4, 0.4]
"car"     -> [-0.3, 0.9, -0.1, 0.2]
"truck"   -> [-0.2, 0.8, -0.2, 0.3]
```

Notice:
- "cat" and "dog" have similar numbers — they're both animals
- "car" and "truck" have similar numbers — they're both vehicles
- "cat" and "car" have very different numbers — they're unrelated

### The magic: nobody hand-coded these numbers

Nobody sat down and said "dimension 1 should represent animal-ness." The model discovered these patterns on its own during training, by reading billions of sentences and adjusting the numbers to reduce prediction errors.

This is a deep point worth pausing on:

The model was given random numbers for each token at the start. Through training on enormous amounts of text, those random numbers gradually shifted until tokens that appear in similar contexts ended up with similar numbers.

### Why this works: the distributional hypothesis

There's a foundational idea in linguistics:

> "You shall know a word by the company it keeps."

Words that appear in similar contexts tend to have similar meanings.

- "The **cat** sat on the mat."
- "The **dog** sat on the mat."
- "The **kitten** sat on the mat."

"cat", "dog", and "kitten" all appear in similar sentence patterns. So the model learns to give them similar embedding vectors.

Meanwhile:

- "The **algorithm** processed the data."
- "The **function** processed the data."

"algorithm" and "function" get similar vectors — different from "cat" and "dog" but close to each other.

### Analogy: GPS coordinates for meaning

Think of embeddings as GPS coordinates, but in a "meaning space" instead of geographic space.

In real GPS:
- New York and New Jersey are close (nearby coordinates)
- New York and Tokyo are far (distant coordinates)
- Closeness means geographic nearness

In embedding space:
- "happy" and "joyful" are close (similar meaning coordinates)
- "happy" and "refrigerator" are far (different meaning coordinates)
- Closeness means semantic nearness

The key difference is that real GPS has 2 dimensions (latitude, longitude). Embedding space might have 768 or more dimensions. More dimensions let the model capture more nuanced relationships.

### What do the individual numbers mean?

Honestly? They're not cleanly interpretable one-by-one.

It's not like:
- Dimension 1 = animal-ness
- Dimension 2 = size
- Dimension 3 = color

It's more like each dimension captures a tiny fragment of many overlapping concepts. The **pattern across all dimensions together** is what encodes meaning. No single number means much alone.

Think of it like a face. No single pixel defines a face. But the pattern of all pixels together creates a recognizable identity.

### How is the embedding table structured?

The model has an embedding matrix — think of it as a giant lookup table:

```
Token ID 0    -> [0.12, -0.34, 0.56, ...]  (768 numbers)
Token ID 1    -> [0.78, 0.23, -0.11, ...]   (768 numbers)
Token ID 2    -> [-0.45, 0.67, 0.89, ...]   (768 numbers)
...
Token ID 49999 -> [0.33, -0.22, 0.44, ...]  (768 numbers)
```

50,000 rows (one per token in vocabulary), each row has 768 numbers. That's the embedding table. When the model sees token ID 5765, it looks up row 5765 and gets that token's embedding vector.

### Important: these embeddings are just the starting point

The embedding you get from this table is the token's **initial** representation — before any context is applied.

At this stage:
- "bank" in "river bank" has the exact same embedding as "bank" in "bank approved loan"
- Context hasn't been applied yet
- That's what attention (covered in the next file) will handle

---

## Positional information: order matters

There's one more piece needed at this stage.

Consider these two sentences:
- "Dog bites man"
- "Man bites dog"

Same tokens, completely different meaning. The difference is **order**.

But embeddings alone don't encode position. If you just look up embeddings for "dog", "bites", "man" — you get the same three vectors regardless of order.

So the model adds **positional information** to each embedding.

### How position is encoded

The model creates a second set of vectors — one for each possible position in the sequence:

```
Position 1 -> [0.01, 0.02, -0.03, ...]
Position 2 -> [0.04, -0.01, 0.05, ...]
Position 3 -> [0.02, 0.03, -0.02, ...]
...
```

These position vectors are combined (added) with the token embeddings:

```
Final input for token at position 1 = token_embedding + position_1_vector
Final input for token at position 2 = token_embedding + position_2_vector
```

Now the model knows both **what** the token is and **where** it sits in the sequence.

### Analogy

Imagine students sitting in a classroom. The embedding is their name badge (who they are). The position encoding is their seat number (where they sit). The model needs both to understand the sentence.

---

## Putting it together: what the model sees after this stage

For the sentence: **"The animal didn't cross the street"**

```
Step 1 - Tokenize:
  "The"     -> ID 464
  "animal"  -> ID 5765
  "didn't"  -> ID 1422
  "cross"   -> ID 3272
  "the"     -> ID 464
  "street"  -> ID 4891

Step 2 - Look up embeddings:
  ID 464   -> [0.12, -0.34, 0.56, ..., 0.23]  (768 numbers)
  ID 5765  -> [0.78, 0.23, -0.11, ..., 0.45]  (768 numbers)
  ... and so on for each token

Step 3 - Add position:
  Token "The" at position 1:
    [0.12, -0.34, ...] + [0.01, 0.02, ...] = [0.13, -0.32, ...]
  Token "animal" at position 2:
    [0.78, 0.23, ...] + [0.04, -0.01, ...] = [0.82, 0.22, ...]
  ... and so on
```

The result is a set of vectors — one per token — that encode both identity and position. These vectors are the **input** to the first transformer layer.

At this point, no context has been applied. "animal" doesn't yet know it's followed by "didn't cross." That's what the next stage (attention) handles.

---

## Summary

| Stage | What happens | Analogy |
|-------|-------------|---------|
| Tokenization | Text split into subword pieces, each gets an ID | Assigning roll numbers to students |
| Embedding lookup | Each ID gets a learned vector of 768+ numbers | Each student gets a detailed profile card |
| Position encoding | Order information is added to each vector | Each student is also given a seat number |
| Result | Set of vectors ready for the transformer layers | Classroom of students with profiles and seats, ready for discussion |

---

**Next: [02 — Attention](./02_attention.md)** — how each token decides which other tokens matter for its meaning.
