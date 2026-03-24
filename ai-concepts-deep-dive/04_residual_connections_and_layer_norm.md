# Residual Connections and Layer Normalization

## Where we left off

We now know three components:
- **Embeddings** give tokens their initial identity ([01](./01_tokens_and_embeddings.md))
- **Attention** lets tokens gather context from each other ([02](./02_attention.md))
- **MLP** refines each token's representation after gathering ([03](./03_mlp_feed_forward.md))

A transformer stacks many layers of attention + MLP. But here's a problem that isn't obvious: **stacking many layers can break things**. The numbers inside the model can become wildly large, wildly small, or lose important earlier information.

Residual connections and layer normalization exist to prevent this.

---

## Part 1: Residual connections

### The problem they solve

Imagine you're renovating a house. You tear down a wall, rebuild it, tear down another, rebuild it. After 50 rounds of tearing down and rebuilding, the house might look nothing like the original — and some of the good structural work from early rounds might be completely lost.

The same thing happens in a deep neural network. Each layer transforms the data. After 24, 48, or 96 layers of transformation, the original input information can get distorted beyond recovery.

Even worse: during training, the model learns by sending error signals backward through all layers. If those signals must travel through 96 layers of transformations, they can shrink to near-zero (vanishing) or explode to huge numbers (exploding). Either way, learning breaks.

### What a residual connection does

A residual connection is remarkably simple. Instead of replacing the old representation with the new one, it **adds** the new to the old.

Here's the idea:

```
Without residual:
    input -> [attention] -> output
    (old information is replaced)

With residual:
    input -> [attention] -> attention_output
    final = input + attention_output
    (old information is preserved, new information is added on top)
```

### Analogy: editing a document with track changes

Imagine writing a report.

**Without residual (destructive editing):**  
You write a draft, then your colleague completely rewrites it. Then another colleague completely rewrites that. After 10 rewrites, nothing from your original draft survives.

**With residual (track changes):**  
You write a draft. Your colleague adds suggestions **on top of** your original text. The next colleague adds more suggestions on top of that. After 10 rounds, your original text is still there, with layers of improvements stacked on top.

If any single round of editing made things worse, the original is still preserved. The model can learn to make small, useful additions at each layer rather than risky complete rewrites.

### Why this matters for deep networks

With residual connections:
- Information from the initial embedding can survive all the way to the final layer
- Each layer only needs to learn what to **add** (the "residual" — hence the name)
- Error signals during training can flow backward through the skip path directly, avoiding the vanishing problem
- The model becomes much easier to train with many layers

Without residual connections, training a 96-layer transformer would be nearly impossible. With them, it works reliably.

### Where residual connections appear

In each transformer layer, there are **two** residual connections:

```
1. Around attention:
   enriched = original + attention(original)

2. Around MLP:
   refined = enriched + mlp(enriched)
```

Both attention and MLP have skip paths, so both the gathering step and the thinking step preserve earlier information.

---

## Part 2: Layer normalization

### The problem it solves

As data flows through many layers, the numbers in token vectors can drift in unpredictable ways:
- Some dimensions might become very large (thousands)
- Others might become very small (near zero)
- The overall scale might shift from layer to layer

This makes learning unstable. The model's internal adjustments during training become unreliable when the numbers it's working with keep changing in scale.

### Analogy: standardizing test scores

Imagine a school with 30 teachers, each giving tests. Teacher A gives scores out of 100. Teacher B gives scores out of 50. Teacher C gives scores out of 1000.

If you try to compare or combine these scores directly, it's chaos. A student with 45 from Teacher B is doing great (90%), but a student with 45 from Teacher A is average (45%), and 45 from Teacher C is terrible (4.5%).

The solution: **normalize** every teacher's scores to the same scale. Convert all scores to a 0-100 scale, or to a "how many standard deviations from the class average" scale.

Now you can combine and compare fairly.

Layer normalization does this for the numbers inside token vectors at each layer.

### What layer normalization actually does

For each token's vector, it:

1. **Calculates the average** of all the numbers in that vector
2. **Calculates how spread out** the numbers are (standard deviation)
3. **Adjusts each number** so the average becomes 0 and the spread becomes 1
4. **Then applies learned scaling and shifting** so the model can fine-tune the exact range that works best

### Why this helps

After normalization:
- Numbers stay in a predictable range
- Each layer receives input in a consistent format
- Training is more stable because the model isn't chasing moving targets
- Layers can focus on learning useful patterns instead of adapting to wildly varying input scales

### Where layer normalization appears

In modern transformers, layer normalization typically appears:

```
1. Before attention:
   normalized = layer_norm(input)
   attended = input + attention(normalized)

2. Before MLP:
   normalized = layer_norm(attended)
   output = attended + mlp(normalized)
```

So the data is normalized before each major processing step, ensuring that both attention and MLP receive well-behaved inputs.

---

## How residual connections and layer normalization work together

They're a team. Here's why both are needed:

**Residual connections alone:**  
Information is preserved, but numbers can still drift in scale over many layers. Layer 50 might be dealing with numbers 100x larger than layer 1. Learning becomes unstable.

**Layer normalization alone:**  
Numbers stay well-scaled, but without residuals, each layer completely overwrites the previous one. Deep training still struggles because error signals degrade.

**Together:**  
- Residual connections preserve information and provide clean paths for error signals
- Layer normalization keeps the numbers well-behaved at every step
- The combination allows stable training of very deep networks (50+ layers)

### Analogy: highway with speed limits

Think of a multi-lane highway:
- **Residual connection** = the highway itself, letting traffic (information) flow continuously from start to finish without being forced through every local street
- **Layer normalization** = speed limits at regular intervals, keeping traffic flowing at a manageable, consistent pace

Without the highway, traffic gets stuck in local streets (information loss). Without speed limits, traffic becomes chaotic (numerical instability). Both together create smooth, reliable flow.

---

## Why you should care about these "boring" components

Residual connections and layer normalization are not glamorous. They don't get the headlines that attention gets. But without them:

- Transformers couldn't have more than a few layers
- Training would fail on large models
- The rich, deep understanding that comes from 24-96 layers of refinement would be impossible

They're the plumbing and electrical wiring of the building. Nobody talks about them, but the building doesn't function without them.

---

## Summary

| Component | Problem it solves | How it works | Analogy |
|-----------|------------------|-------------|---------|
| Residual connection | Information loss over many layers | Adds new output to old input instead of replacing | Track changes on a document — additions, not rewrites |
| Layer normalization | Numbers drifting to extreme scales | Rescales each vector to consistent range | Standardizing test scores across teachers |
| Together | Enables stable training of deep networks | Preservation + stability = reliable depth | Highway with speed limits |

---

**Previous: [03 — MLP / Feed-Forward Block](./03_mlp_feed_forward.md)**  
**Next: [05 — Transformer Block and Layers](./05_transformer_block_and_layers.md)** — putting all pieces together into one complete system.
