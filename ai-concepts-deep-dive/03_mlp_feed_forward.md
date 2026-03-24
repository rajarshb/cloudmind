# MLP / Feed-Forward Block

## Where we left off

After attention ([02](./02_attention.md)), each token has gathered relevant information from other tokens. The token "it" in our sentence now carries context like "I probably refer to the animal, which is tired."

But gathering information is only half the job. The other half is: **what do you do with the gathered information?**

That's what the MLP block handles.

---

## The problem the MLP solves

Attention is like collecting ingredients. MLP is like cooking.

Let's make this concrete.

Suppose you're writing a book report on a mystery novel. You've done your research:
- You gathered notes from chapters 1-10
- You highlighted key plot twists
- You marked character relationships

At this point, you have raw material. But you haven't **thought about it** yet. You haven't combined "the butler was absent in chapter 3" with "the poison was bought in chapter 5" to conclude "the butler might be the killer."

That combination — taking gathered evidence and producing deeper understanding — is what the MLP does for each token.

---

## What MLP stands for

MLP = Multi-Layer Perceptron. That name is historical and honestly not helpful for understanding. Forget the name. Think of it as:

> A per-token thinking block that takes the context-enriched representation from attention and transforms it into a more useful representation.

---

## How MLP works (concept, not formula)

The MLP is actually quite simple in structure. It does two things, back to back:

### Step 1: Expand

The token's current vector (say 768 numbers) is expanded into a larger space (say 3072 numbers — typically 4x the original size).

Why expand? Because in a bigger space, there's more room to detect and represent complex patterns.

### Analogy: microscope

Think of a scientist examining a cell. With the naked eye (768 dimensions), they see basic shapes. Under a microscope (3072 dimensions), they see internal structures, membranes, organelles — much more detail.

The expansion step is like putting the token under a microscope.

### Step 2: Activate (decide what matters)

After expanding, the model applies an "activation" — a filtering step. This decides which of those expanded features are important and which should be suppressed.

Some expanded features will be strong (these patterns are present and relevant). Others will be near zero (these patterns aren't useful right now).

### Analogy: highlighting a textbook

You've photocopied a page and enlarged it (expansion). Now you go through with a highlighter and mark what matters, leaving the rest unhighlighted (activation). Only the highlighted parts will influence your final understanding.

### Step 3: Compress back

After expansion and activation, the result is compressed back to the original size (768 numbers).

Now the token has the same shape as before, but its representation has been **refined** — the useful patterns have been amplified, the noise has been dampened.

---

## Why can't attention alone do this?

This is a crucial question.

Attention gathers information by **mixing** representations from other tokens. But mixing is just weighted averaging — it's a **linear** operation. It can blend existing information, but it cannot create fundamentally new features.

Think about it with colors:
- Mixing blue and yellow gives green. That's useful.
- But mixing can never give you a texture, a pattern, or a gradient. It just blends what already exists.

The MLP introduces **nonlinearity** — the ability to create new patterns that didn't exist in any individual token's representation.

Going back to the detective analogy:
- Attention = gathering clues from different parts of the crime scene
- MLP = **reasoning about the clues** — combining "muddy shoes" + "broken window" into the new insight "intruder entered through window from outside"

That new insight ("intruder entered from outside") didn't exist in any single clue. It was created by processing multiple clues together. That's what nonlinear transformation does.

---

## What "per-token" means and why it matters

A critical detail: MLP processes each token **independently**. It doesn't look at other tokens. That's attention's job.

After attention, "it" has a context-enriched vector. The MLP takes only that vector and refines it. It doesn't look at "animal" or "tired" again during this step — that information is already baked into "it"'s vector from the attention step.

### Why is this separation important?

It creates a clean division of labor:
- **Attention** handles inter-token communication (who talks to whom)
- **MLP** handles intra-token computation (what each token does with the information it received)

This separation makes the system modular and effective. Each part does one thing well.

---

## Where does knowledge live?

Research has shown that the MLP blocks are where a lot of the model's **factual knowledge** gets stored.

During training, the model reads billions of sentences containing facts like "Paris is the capital of France." The MLP weights gradually encode these patterns.

When the model later processes a sentence containing "the capital of France," the MLP activations help surface the associated concept "Paris" from the learned weight patterns.

So attention routes information, but MLP is a major place where stored knowledge lives and gets applied.

---

## Real-world analogy: post office

Think of a post office as a transformer layer:

- **Attention** = the sorting system that routes each letter to the right recipient, combining relevant mail together
- **MLP** = the recipient reading their mail and writing a response

The sorting system (attention) doesn't read or interpret the mail. It just delivers it to the right place. The recipient (MLP) reads the delivered mail and produces something new from it.

Both steps are essential. Sorting without reading is useless. Reading without proper delivery is chaotic.

---

## After MLP: what the token looks like

Before attention + MLP (start of layer):
- "it" = generic pronoun with position info

After attention:
- "it" = pronoun that has gathered context about animal, tiredness

After MLP:
- "it" = representation that has **processed** that context into deeper features, such as "animate referent in fatigued state"

The shape of the vector hasn't changed (still 768 numbers), but the **quality and depth of information** in those numbers has improved.

---

## Summary

| Concept | What it does | Analogy |
|---------|-------------|---------|
| Expansion (step 1) | Increases vector size to detect more patterns | Microscope revealing more detail |
| Activation (step 2) | Filters to keep useful patterns, suppress noise | Highlighting a textbook page |
| Compression (step 3) | Returns to original vector size with refined information | Writing a concise summary from highlighted notes |
| Overall MLP role | Per-token reasoning and transformation after attention | Reading and interpreting delivered mail |
| Why needed | Attention mixes but can't create new patterns; MLP can | Mixing paint gives new colors; cooking creates new dishes |

---

**Previous: [02 — Attention](./02_attention.md)**  
**Next: [04 — Residual Connections and Layer Normalization](./04_residual_connections_and_layer_norm.md)** — the hidden infrastructure that makes deep stacking possible.
