# Training and Learning

## Where we left off

We now understand the complete transformer architecture — tokens, embeddings, attention, MLP, residuals, layer norm, and the encoder/decoder variants. But we've been treating the model as if it already "knows" things.

How did it learn? Where did the knowledge come from? Why do the attention weights, MLP weights, and embedding vectors have the values they have?

The answer is **training** — and understanding training is essential because it explains **why** the model behaves the way it does, and also **why** it sometimes fails.

---

## The core idea: learning from mistakes

Training a neural network is conceptually simple:

1. Show the model some input
2. Ask it to make a prediction
3. Compare the prediction to the correct answer
4. Measure how wrong the prediction was
5. Adjust the model slightly to be less wrong next time
6. Repeat billions of times

That's it. The entire intelligence of GPT-4, Claude, or any large language model comes from this loop repeated on enormous amounts of text.

---

## Let's make this concrete with a decoder model (like GPT)

### The training task: predict the next word

The model is given a piece of text and must predict the next token at every position.

Training sentence:  
**"The animal didn't cross the street because it was too tired."**

The model sees this as a series of prediction challenges:

```
Given: "The"
Predict: "animal"

Given: "The animal"
Predict: "didn't"

Given: "The animal didn't"
Predict: "cross"

Given: "The animal didn't cross"
Predict: "the"

Given: "The animal didn't cross the"
Predict: "street"

... and so on for every position.
```

Each of these is a separate prediction that the model must get right.

### What "predict" means in practice

The model doesn't output a single word. It outputs a **probability for every token in its vocabulary** (all 50,000 of them).

For the input "The animal didn't cross the":

```
Model's output (probabilities):
  "street"    -> 0.15  (15% confidence)
  "road"      -> 0.12  (12% confidence)
  "bridge"    -> 0.05  (5% confidence)
  "ball"      -> 0.001 (0.1% confidence)
  "the"       -> 0.002 (0.2% confidence)
  ... 49,995 other tokens with various tiny probabilities
```

The correct answer is "street." The model gave it 15% — which is the highest, so it would have predicted correctly. But 15% is not very confident. The model should ideally be much more confident about "street."

---

## Step-by-step: how one training example works

### Step 1: Forward pass (make the prediction)

The input text flows through the entire transformer:
- Tokenization
- Embedding + position
- Through all transformer layers (attention + MLP in each)
- Output head produces probability distribution over vocabulary

This is called the **forward pass** — data moves forward through the network.

### Step 2: Calculate the loss (measure the mistake)

The **loss** is a single number that says "how wrong was the prediction?"

If the correct next token is "street" and the model assigned it 15% probability, that's a moderate loss. If the model assigned 95% probability, the loss would be very low. If the model assigned 0.01% probability, the loss would be very high.

The loss is calculated for every prediction in the sentence, then averaged.

### Analogy: classroom quiz with instant grading

Think of a student taking a multiple-choice quiz where each question has 50,000 options.

- Question: "The animal didn't cross the ___"
- Student's answer distribution: somewhat confident about "street" but also considering "road" and "bridge"
- Correct answer: "street"
- Grade for this question: moderate (student got it right but wasn't confident enough)

The loss is like the inverse of the grade. Low loss = good prediction. High loss = bad prediction.

### Step 3: Backward pass (figure out who's responsible)

This is where the magic happens. The model needs to figure out: **which of its billions of internal numbers should change, and in which direction, to reduce the loss?**

The process is called **backpropagation** — the error signal flows backward through the network.

Think of it like this: the loss is traced back through every computation that led to the wrong answer.

```
Wrong prediction at the output
  ↓ blame flows backward
Output head weights (slightly responsible)
  ↓
Layer 12 MLP weights (somewhat responsible)
  ↓
Layer 12 Attention weights (somewhat responsible)
  ↓
...
  ↓
Layer 1 Attention weights (slightly responsible)
  ↓
Embedding table values (slightly responsible)
```

Every single learned number in the model gets a tiny nudge direction: "increase a little" or "decrease a little" to make the prediction better next time.

### Analogy: tracing a factory defect

A car comes off the assembly line with a rattling noise.

The quality inspector traces backward:
- Was it the final assembly? Partly — a bolt wasn't tight enough.
- Was it the engine fitting? Partly — alignment was slightly off.
- Was it the parts supplier? Partly — one component was 0.1mm too large.

Each station gets specific feedback: "adjust this by this much." No single station is fully responsible, but each contributes to the final problem.

Backpropagation does exactly this — traces the prediction error back through every component and assigns responsibility.

### Step 4: Update weights (make the adjustment)

Based on the backward pass, every learned number in the model is adjusted by a tiny amount.

The adjustment size is controlled by a value called the **learning rate**. Too large, and the model overshoots (fixes one mistake but creates others). Too small, and learning takes forever.

### Analogy: tuning a guitar

If a string is flat, you tighten it slightly. If it's sharp, you loosen it. You don't crank the tuning peg a full turn — that would overshoot. You make small, careful adjustments.

The learning rate is how much you turn the peg. Too much and you go from flat to sharp. Too little and you're turning all day. Just right and you gradually approach perfect pitch.

### Step 5: Repeat with the next batch of text

One training example barely moves the needle. The model needs to see the same patterns in thousands of different contexts before it reliably learns them.

So training repeats this loop on millions of text passages, each time making tiny adjustments. Over time, the adjustments accumulate into genuine language understanding.

---

## What exactly gets learned during training?

Every number that the model can adjust is called a **parameter**. Modern large models have billions of them:

- **Embedding table values:** which numbers represent each token's initial identity
- **Attention weight matrices:** what kinds of question cards, label cards, and content cards to create
- **MLP weight matrices:** how to refine token representations
- **Layer norm parameters:** fine-tuning of the normalization scaling
- **Output head weights:** how to convert final representations into vocabulary probabilities

All of these start as random numbers before training. After training on billions of tokens, they hold patterns that encode grammar, facts, reasoning heuristics, and more.

### Where knowledge lives

A common question: "Where is the knowledge stored?"

The answer: it's distributed across all parameters, with different components playing different roles.

- **Embedding table:** word identities and basic semantic groupings
- **Attention weights:** patterns for how words relate to each other (grammar, coreference, dependency)
- **MLP weights:** factual associations and complex feature combinations
- **Later layers:** more abstract, task-relevant patterns
- **Earlier layers:** more basic, syntactic patterns

No single parameter stores a fact like "Paris is the capital of France." Instead, this knowledge is encoded as a pattern across many parameters that, when activated together by the right input, produce the right output.

---

## Training data: where the knowledge comes from

The model learns from the text it's trained on. This is critically important to understand.

### What training data looks like

For large language models, training data typically includes:
- Books (fiction and non-fiction)
- Wikipedia
- Web pages (filtered for quality)
- Code repositories
- Academic papers
- News articles
- Forum discussions

Collectively, this might be trillions of tokens of text.

### What the model actually learns from this data

By predicting the next token across all this text, the model implicitly learns:

- **Grammar:** "she goes" is far more common than "she go," so the model learns subject-verb agreement
- **Facts:** "The capital of France is Paris" appears in many forms across the data, so the model associates France's capital with Paris
- **Reasoning patterns:** math problems with solutions, logical arguments with conclusions — the model picks up reasoning traces
- **Style and tone:** formal writing vs casual writing vs technical writing — the model learns to match patterns

### What the model does NOT learn

- It doesn't learn a database of facts that it can query like SQL
- It doesn't learn rules that it consciously follows
- It doesn't understand in the way humans understand — it discovers statistical regularities

This is why models can sometimes confidently say wrong things. They've learned a pattern that usually works but isn't always correct. The pattern "the capital of [country] is [city]" was learned from data, not from a verified database.

---

## Training for encoder models (like BERT — relevant to embeddings)

Decoder models learn by predicting the next token. Encoder models learn differently.

### Masked Language Modeling (MLM)

The main training task for BERT-style models:

1. Take a sentence
2. Randomly hide (mask) some tokens
3. Ask the model to predict the hidden tokens using the surrounding context

Example:

```
Original: "The animal didn't cross the street because it was too tired."
Masked:   "The animal [MASK] cross the street because it was [MASK] tired."

Model must predict:
  [MASK] at position 3 = "didn't"
  [MASK] at position 9 = "too"
```

Because the model can see both left and right context (bidirectional), it learns very rich understanding of language.

### Why this matters for embeddings

MLM training forces the model to deeply understand context from all directions. This produces token representations that are highly informative about meaning. When these representations are pooled into a single sentence vector, that vector captures the full meaning of the sentence — which is exactly what we need for semantic search.

---

## Contrastive learning (training specifically for embeddings)

For semantic search, there's an additional training method that's crucial to understand.

### The problem with MLM alone

MLM produces good general-purpose representations, but they're not optimized for comparing sentence-level meaning. Two sentences with similar meaning might not have similar vectors.

### The solution: contrastive learning

Contrastive learning trains the model with pairs of sentences:

**Positive pair (similar meaning):**
- "How do I reduce my cloud storage costs?"
- "Ways to lower S3 billing expenses"

**Negative pair (different meaning):**
- "How do I reduce my cloud storage costs?"
- "Best Italian restaurants in Chicago"

The training objective:
- Make embeddings of the positive pair **close together**
- Make embeddings of the negative pair **far apart**

### Analogy: organizing a library

Imagine you're organizing books in a library. You have unlimited shelf space.

Contrastive learning is like a librarian who:
- Picks up "Introduction to Python" and "Python for Beginners" and says "these should be on the same shelf" (push vectors closer)
- Picks up "Introduction to Python" and "Italian Cooking" and says "these should be on different shelves" (push vectors apart)

After doing this for millions of book pairs, the library is organized by meaning. Similar topics are near each other. Unrelated topics are far away. Now when someone brings in a new book, the librarian knows exactly where it belongs based on its content.

This is exactly what happens in embedding space after contrastive training.

### Why contrastive learning produces the numbers like [0.12, -0.87, 0.43, ...]

Those specific numbers are the result of billions of parameter adjustments driven by contrastive loss:

- Every time two similar sentences were pushed closer in vector space, parameters shifted slightly
- Every time two different sentences were pushed apart, parameters shifted slightly
- After billions of such adjustments, the model produces vectors where the geometry (distances and directions) reflects semantic meaning

The numbers are not hand-designed. They're the end product of a massive optimization process.

---

## How much training does it take?

For a large language model:
- **Data:** trillions of tokens
- **Compute:** thousands of GPUs running for weeks or months
- **Cost:** millions of dollars
- **Parameters adjusted:** billions, each adjusted millions of times

For an embedding model (smaller but specialized):
- **Data:** millions to billions of text pairs
- **Compute:** hundreds of GPUs for days
- **Cost:** thousands to hundreds of thousands of dollars
- **Parameters:** hundreds of millions

---

## Summary

| Concept | What it is | Analogy |
|---------|-----------|---------|
| Forward pass | Input flows through model, prediction is made | Student takes a quiz |
| Loss | Measure of how wrong the prediction was | Quiz grade (inverted) |
| Backward pass | Error signal traced back through all components | Factory defect trace |
| Weight update | Each parameter adjusted slightly | Fine-tuning a guitar string |
| Training loop | Forward + loss + backward + update, repeated billions of times | Practicing a skill daily for years |
| MLM (encoder training) | Predict masked tokens using surrounding context | Fill-in-the-blank exercise |
| Next-token prediction (decoder training) | Predict next word from prior words | Completing someone's sentence |
| Contrastive learning (embedding training) | Push similar pairs closer, dissimilar pairs apart | Librarian organizing books by topic |
| Parameters | All learnable numbers in the model | All tunable knobs in a complex machine |

---

**Previous: [07 — Encoder vs Decoder](./07_encoder_vs_decoder.md)**  
**Next: [09 — Vector Embeddings and Semantic Search](./09_vector_embeddings_and_semantic_search.md)** — connecting transformer representations to search systems.
