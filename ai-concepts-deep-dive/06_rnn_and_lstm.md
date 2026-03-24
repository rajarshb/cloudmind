# RNN and LSTM

## Why learn about older approaches?

Transformers are the current standard, but RNN and LSTM came first. Understanding them does two things:

1. Shows you **why** transformers were invented — what problems needed solving
2. Gives you a deeper appreciation of what attention actually changed

If you only learn transformers, you know the "what" but not the "why." This file gives you the "why."

---

## Part 1: RNN (Recurrent Neural Network)

### The problem RNN was built to solve

Before RNNs, early neural networks had a fundamental limitation: they could only process **fixed-size inputs**.

Give it an image of 100x100 pixels? Fine — 10,000 inputs, always.  
Give it a sentence? Problem. Sentences have different lengths. "Hi" is 1 word. "The animal didn't cross the street because it was too tired" is 11 words.

You can't build a fixed-size system for variable-length input. Language needs a model that can handle any length.

### The RNN idea: process one token at a time, carry forward a memory

RNN's solution is elegant in concept:

1. Read token 1. Create a memory note.
2. Read token 2. Update the memory note using token 2 + the old note.
3. Read token 3. Update the memory note using token 3 + the old note.
4. Continue until the sentence ends.
5. Use the final memory note as the "understanding" of the sentence.

At every step, the model has two inputs:
- The current token
- The memory from everything before

And it produces one output:
- An updated memory

### Analogy: listening to a story on the phone

You're on a phone call. Someone is telling you a story, one sentence at a time. You can't replay what they said. You can't jump back. You can only listen to the current sentence and try to keep a running mental summary.

After the first sentence: you have a rough idea.  
After the fifth sentence: your mental summary is richer.  
After the twentieth sentence: your summary is dense, but you might have forgotten details from sentence 1.

That running mental summary is the RNN's hidden state.

### Let's trace our sentence through an RNN

**"The animal didn't cross the street because it was too tired."**

```
Step 1: Read "The"
  Memory: [something about an article starting a phrase]

Step 2: Read "animal"
  Memory: [there's an animal being discussed]

Step 3: Read "didn't"
  Memory: [the animal didn't do something]

Step 4: Read "cross"
  Memory: [the animal didn't cross something]

Step 5: Read "the"
  Memory: [the animal didn't cross something specific]

Step 6: Read "street"
  Memory: [the animal didn't cross the street]

Step 7: Read "because"
  Memory: [the animal didn't cross the street, and a reason follows]

Step 8: Read "it"
  Memory: [something about "it" — but what does "it" refer to?
           The model must rely on whatever traces of "animal" and
           "street" survived in the compressed memory from steps 2 and 6]

Step 9: Read "was"
  Memory: [it was something]

Step 10: Read "too"
  Memory: [it was too something]

Step 11: Read "tired"
  Memory: [it was too tired — this is the reason]
```

### The critical weakness: memory compression

Notice what happened at Step 8. When the model reaches "it," the information about "animal" from Step 2 has been through **6 memory updates**. Each update partially overwrites the previous memory.

It's like playing the telephone game. Person 1 says something. Person 2 repeats it to Person 3. By Person 7, the original message is distorted.

The RNN's memory is a fixed-size vector (a list of, say, 256 or 512 numbers). Everything the model has read must be compressed into this fixed-size container. As the sentence gets longer, earlier information gets increasingly squeezed and potentially lost.

### The vanishing gradient problem (conceptual explanation)

During training, the model needs to learn from its mistakes. If the model makes a bad prediction about "it" at step 8, it needs to send an error signal back to step 2 (where "animal" was processed) to learn "you should have remembered 'animal' better."

But that error signal must travel backward through 6 steps. At each step, the signal gets multiplied by certain factors. If those factors are less than 1 (which they often are), the signal shrinks exponentially:

```
Step 8: error signal = 1.0
Step 7: signal = 1.0 × 0.7 = 0.7
Step 6: signal = 0.7 × 0.7 = 0.49
Step 5: signal = 0.49 × 0.7 = 0.34
Step 4: signal = 0.34 × 0.7 = 0.24
Step 3: signal = 0.24 × 0.7 = 0.17
Step 2: signal = 0.17 × 0.7 = 0.12
```

By the time the signal reaches step 2, it's tiny. The model barely learns from the connection between "animal" and "it." For sentences of 50 or 100 words, the signal essentially vanishes.

This is why it's called the **vanishing gradient** problem. Gradient is just the technical term for this error signal.

### Why RNN still mattered historically

Despite these weaknesses, RNNs were groundbreaking when introduced. They showed that neural networks could handle sequential data at all — a major step forward from fixed-input networks. They were used successfully for:
- Simple language tasks
- Short sequences
- Time-series prediction
- Speech recognition (early versions)

But for anything requiring long-range understanding, they struggled.

---

## Part 2: LSTM (Long Short-Term Memory)

### The problem LSTM was built to solve

LSTM was specifically designed to fix the vanishing gradient problem of RNNs. The goal: build a model that can remember important information over long sequences.

### The key insight: add a separate long-term memory highway

An RNN has one memory (hidden state) that gets completely transformed at every step. LSTM adds a second memory — the **cell state** — that serves as a protected long-term memory highway.

Think of it this way:

**RNN:** One lane road. Every car (piece of information) must merge through this single lane. Traffic jams. Cars get lost.

**LSTM:** Two roads. A local street (hidden state) for current, active processing. And a highway (cell state) where important information can travel long distances without being forced through every local intersection.

### The three gates (the core innovation)

LSTM controls what goes into and out of the long-term memory using three "gates." A gate is just a learned switch that opens or closes to control information flow.

### Gate 1: The Forget Gate

**Question it answers:** "What old information should I throw away?"

At each step, the forget gate looks at the current token and the previous state, and decides which parts of the long-term memory to keep and which to discard.

**Analogy:** You're packing for a trip. At each stop, you decide: do I still need this item in my suitcase? Keep the passport (always relevant). Throw away the city map from the previous city (no longer relevant).

**Example in our sentence:**
When the model reads "because" at step 7, the forget gate might decide:
- Keep "animal" information (likely still relevant for what follows)
- Keep "didn't cross the street" (the main event)
- Reduce "The" information (article, no longer important)

### Gate 2: The Input Gate

**Question it answers:** "What new information should I store in long-term memory?"

Not everything the model reads is worth remembering long-term. The input gate filters what's important enough to add to the cell state.

**Analogy:** You're taking notes during a lecture. Not everything the professor says goes into your notes. You filter — recording key concepts and skipping filler words and tangents.

**Example in our sentence:**
When the model reads "tired" at step 11:
- Input gate says: "tired" is highly important — it's the reason for the main event
- This gets stored strongly in long-term memory
- The model can now connect "tired" with "animal" (which was preserved by the forget gate)

### Gate 3: The Output Gate

**Question it answers:** "What part of my long-term memory should I expose right now?"

Not all stored memory is relevant at every moment. The output gate controls what portion of the cell state to use for the current step.

**Analogy:** You have a filing cabinet with hundreds of folders. When answering a specific question, you don't dump all folders on the desk. You pull out only the relevant ones.

**Example in our sentence:**
When the model processes "it" at step 8:
- Long-term memory contains information about "animal", "street", "didn't cross"
- Output gate decides: right now, noun candidates are most relevant
- It exposes "animal" and "street" features for the current processing, keeps other stored info filed away

### How this improves over RNN

The cell state highway allows information to flow across many steps with minimal transformation. The forget gate learns to keep important information alive. The error signal during training can flow back through the cell state path without degrading as badly.

Let's revisit the error signal example:

```
RNN error signal at step 2: 0.12 (nearly vanished)

LSTM error signal at step 2: 0.85 (much stronger)
(because the cell state highway gives the signal a clean path)
```

This means LSTM can actually learn long-distance dependencies that RNN cannot.

### LSTM limitations (why transformers still replaced it)

LSTM was a huge improvement over RNN, but it still has fundamental constraints:

**1. Still sequential.**  
LSTM must read token 1 before token 2, token 2 before token 3, and so on. You can't process multiple tokens at the same time. On modern GPUs that can do thousands of operations in parallel, this is a massive waste.

Training an LSTM on a large dataset is like having a 1000-lane highway but forcing all cars to use a single lane, one at a time.

**2. Still compressed memory.**  
Even with the cell state highway, all past information must fit into a fixed-size vector. For very long documents (thousands of tokens), compression losses still occur. The gates help, but they can't fully overcome the fixed-capacity constraint.

**3. No direct access to distant tokens.**  
When processing "it" at step 8, the LSTM cannot directly look back at "animal" at step 2. It can only access what survived in the cell state through 6 gate operations. The connection is **indirect**.

In a transformer, "it" can directly attend to "animal" in one step, regardless of distance. The connection is **direct**.

---

## Part 3: Direct comparison with our sentence

**"The animal didn't cross the street because it was too tired."**

### How RNN handles "it"

- Reads sequentially to step 8
- At "it," relies on compressed hidden state
- "animal" info may be faded after 6 updates
- Weak at resolving the reference correctly
- If sentence were 50 words instead of 11, resolution would be even harder

### How LSTM handles "it"

- Also reads sequentially to step 8
- At "it," has both hidden state AND cell state
- Forget gate may have preserved "animal" in cell state
- Better chance of correct resolution
- Still limited by indirect access and compression

### How Transformer handles "it"

- Processes all tokens at once in each layer
- At "it," directly computes attention scores to all other tokens
- Scores "animal" high, scores "tired" high (supporting evidence)
- Multiple heads capture different relationship types
- Multiple layers refine the resolution
- Direct access, no compression, parallel processing

---

## Summary table

| Aspect | RNN | LSTM | Transformer |
|--------|-----|------|-------------|
| Processing order | One token at a time | One token at a time | All tokens at once per layer |
| Memory | Single rolling hidden state | Hidden state + cell state (highway) | No rolling memory; direct attention |
| Long-range connections | Weak (telephone game effect) | Better (highway preserves info) | Strong (direct token-to-token attention) |
| Training speed | Slow (sequential) | Slow (sequential) | Fast (parallel on GPU) |
| Core innovation | First sequential neural net | Gates controlling memory flow | Direct attention between all tokens |
| Weakness | Forgets early information | Still sequential, still compressed | Expensive for very long sequences |
| Mental model | Listening to a story on the phone | Same but with better note-taking rules | Everyone in the room at once, anyone can talk to anyone |

---

**Previous: [05 — Transformer Block and Layers](./05_transformer_block_and_layers.md)**  
**Next: [07 — Encoder vs Decoder](./07_encoder_vs_decoder.md)** — the different ways transformers are structured for different tasks.
