# Encoder vs Decoder

## Where we left off

We now understand the full transformer block: tokenize, embed, stack layers of attention + MLP. But there's a design choice that changes how the transformer is used:

**Should each token see the full sentence? Or only what came before it?**

This single decision creates three different transformer variants — encoder, decoder, and encoder-decoder. Each is suited for different tasks.

---

## First, understand the two fundamental language tasks

Before understanding encoder vs decoder, you need to understand the two core things we ask AI to do with language:

### Task type 1: Understand text

Given a piece of text, **analyze** it.

Examples:
- Is this email spam or not spam? (classification)
- What is the sentiment of this review? (sentiment analysis)
- Convert this sentence into a meaning vector for search (embedding)
- Find the answer to a question within a document (question answering)

In all of these, you already have the complete text. The model's job is to **understand** it, not create new text.

### Task type 2: Generate text

Given some starting text (or nothing), **create** new text.

Examples:
- Complete this sentence: "The capital of France is ___"
- Write a poem about autumn
- Translate "hello" to French
- Summarize this article

Here, the model's job is to **produce** new tokens one at a time.

These two task types need different rules about what each token is allowed to see.

---

## The key rule: who can see whom?

### For understanding tasks: see everything

If I give you a review — "The food was great but the service was terrible" — and ask "is this positive or negative?", you need to see the **entire** review before answering. Reading only the first half ("The food was great") would give you the wrong answer.

So for understanding tasks, every token should be able to attend to every other token — both before and after it in the sentence.

This is called **bidirectional attention** — each token can look both left and right.

### For generation tasks: see only the past

When generating text, the model predicts one token at a time. When it's deciding the 5th token, the 6th token doesn't exist yet. So it can only look at tokens 1 through 4.

If the model could see future tokens during training, it would be cheating — looking at the answer before guessing. It wouldn't actually learn to predict.

This is called **causal attention** (or masked attention) — each token can only look left, never right.

### Analogy: exam vs creative writing

**Bidirectional (encoder):** You're given a complete essay and asked to grade it. You can read the whole thing — beginning, middle, end — before forming your judgment. Seeing everything helps you understand it better.

**Causal (decoder):** You're writing a story. You can only use what you've written so far to decide what comes next. You can't look ahead because the next part doesn't exist until you write it.

---

## The three architecture variants

### 1. Encoder-only (bidirectional) — BERT family

**What it is:**  
A stack of transformer layers where every token can attend to every other token — both before and after it.

**What it's good for:**  
Understanding tasks. The model reads the complete text and builds rich representations.

**Real-world examples:**  
BERT, RoBERTa, sentence transformers (used for embeddings and search)

**How our sentence works in an encoder:**

```
"The animal didn't cross the street because it was too tired."

Every token can see every other token:

"it" can directly attend to:
  ← "The", "animal", "didn't", "cross", "the", "street", "because"  (before it)
  → "was", "too", "tired"  (after it)

This bidirectional view gives "it" maximum context.
```

Because "it" can see "tired" (which comes after it), the encoder has an advantage for understanding what "it" refers to. It can use both backward context ("animal," "street") and forward context ("tired") simultaneously.

**Limitation:**  
Encoder-only models are not designed to generate text token by token. They're analyzers, not creators.

### 2. Decoder-only (causal) — GPT family

**What it is:**  
A stack of transformer layers where each token can only attend to itself and tokens that came before it. Future tokens are blocked.

**What it's good for:**  
Text generation. The model predicts the next token based on everything so far.

**Real-world examples:**  
GPT-2, GPT-3, GPT-4, Claude, LLaMA, Gemini

**How our sentence works in a decoder:**

```
"The animal didn't cross the street because it was too tired."

Each token can only see what came before:

"The"     sees: [The]
"animal"  sees: [The, animal]
"didn't"  sees: [The, animal, didn't]
"cross"   sees: [The, animal, didn't, cross]
...
"it"      sees: [The, animal, didn't, cross, the, street, because, it]
"was"     sees: [The, animal, didn't, cross, the, street, because, it, was]
"tired"   sees: [The, animal, didn't, cross, the, street, because, it, was, too, tired]
```

Notice: when the decoder processes "it," it **cannot** see "tired" yet (it comes later). So the decoder must resolve "it" using only backward context. This is harder — but during training on billions of sentences, the model learns to do it well.

**Why this design works for generation:**

When the model is generating text and has produced "The animal didn't cross the street because it was too", it needs to predict the next word. At this moment, "tired" genuinely doesn't exist yet. So the causal restriction during training matches the reality during generation.

**Limitation:**  
For pure understanding tasks, the decoder has less context per token (only past, not future) compared to the encoder.

### 3. Encoder-Decoder — T5 family

**What it is:**  
Two transformer stacks connected:
- **Encoder stack:** reads the complete input with bidirectional attention
- **Decoder stack:** generates output token by token with causal attention, but can also attend to the encoder's representation

**What it's good for:**  
Tasks where you read something and produce something else — translation, summarization, question answering with generated answers.

**Real-world examples:**  
T5, BART, the original transformer (from the "Attention Is All You Need" paper)

**Analogy: interpreter at a conference**

Imagine a simultaneous interpreter:
- First, they listen to the **entire** speech in French (encoder). They can mentally revisit any part of the speech.
- Then, they produce the English translation **word by word** (decoder). Each English word is based on the full French speech plus the English words they've already said.

The encoder gives complete understanding of the input. The decoder generates the output incrementally, consulting the encoder's understanding at each step.

**How our sentence works in encoder-decoder (translation example):**

```
Input (French): "L'animal n'a pas traversé la rue parce qu'il était trop fatigué."

Encoder: reads entire French sentence bidirectionally.
         Builds rich representation of every token.

Decoder: generates English translation one token at a time.
  Step 1: generates "The" (consulting encoder's full representation)
  Step 2: generates "animal" (consulting encoder + "The")
  Step 3: generates "didn't" (consulting encoder + "The animal")
  ...and so on until the full English sentence is produced.
```

---

## Cross-attention: how encoder and decoder talk to each other

In the encoder-decoder setup, the decoder has a special kind of attention called **cross-attention**.

In regular self-attention, a token's question card is compared against label cards from the **same** sequence. In cross-attention, the decoder token's question card is compared against label cards from the **encoder's** output.

**Analogy:** In self-attention, a student asks questions to classmates in the same room. In cross-attention, the student asks questions to students in a **different** room (the encoder's room) who have already discussed and understood a different text.

This lets the decoder "consult" the full input understanding while generating output.

---

## Which variant is used for what?

| Task | Best variant | Why |
|------|-------------|-----|
| Text classification | Encoder | Needs full text understanding, no generation |
| Sentiment analysis | Encoder | Same — bidirectional context helps |
| Sentence embeddings for search | Encoder | Needs to capture full meaning of text |
| Chatbot / text generation | Decoder | Generates text one token at a time |
| Code completion | Decoder | Predicts next code token from prior context |
| Translation | Encoder-Decoder | Read source language fully, generate target language |
| Summarization | Encoder-Decoder (or Decoder) | Read full document, generate shorter version |

### Important note about modern trends

In recent years, decoder-only models have become dominant even for tasks traditionally suited to encoders or encoder-decoders. GPT-4 and Claude are decoder-only but can do translation, summarization, classification, and more — because they're so large and well-trained that the causal attention limitation doesn't hurt much in practice.

However, for **embedding and search** tasks specifically, encoder-based models (like sentence transformers) remain the standard choice because bidirectional attention creates better meaning representations.

---

## How this connects to our bigger picture

Remember, our goal is to understand the full chain from text to vector embeddings to semantic search.

For that chain, the critical insight is:

**Encoder-based models** are what produce the embedding vectors used in semantic search. They read the full text bidirectionally, build rich representations through multiple transformer layers, and then those final representations are pooled into a single vector that captures the meaning of the entire input.

We'll trace this exact process in [09 — Vector Embeddings and Semantic Search](./09_vector_embeddings_and_semantic_search.md).

---

## Summary

| Variant | Attention rule | Strength | Weakness | Analogy |
|---------|---------------|----------|----------|---------|
| Encoder-only | Every token sees every other token | Rich understanding of complete text | Cannot generate text | Reading and grading a complete essay |
| Decoder-only | Each token sees only past tokens | Generates text naturally | Less context per token for understanding | Writing a story word by word |
| Encoder-Decoder | Encoder = bidirectional; Decoder = causal + consults encoder | Best for input-to-output tasks | More complex architecture | Interpreter listening fully, then translating incrementally |

---

**Previous: [06 — RNN and LSTM](./06_rnn_and_lstm.md)**  
**Next: [08 — Training and Learning](./08_training_and_learning.md)** — how the model learns all these patterns from data.
