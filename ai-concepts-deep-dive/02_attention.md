# Attention

## Where we left off

After tokenization and embedding (covered in [01](./01_tokens_and_embeddings.md)), each token in the sentence has a vector — a list of numbers representing its identity and position. But these vectors don't know about each other yet.

"bank" has the same vector whether the sentence is "river bank" or "bank approved the loan." Context hasn't been applied.

Attention is how context gets applied.

---

## The problem attention solves

Language is not a bag of independent words. Every word's meaning depends on the words around it.

Consider:

- "I need to **charge** my phone." (charge = supply electricity)
- "The judge will **charge** the defendant." (charge = formally accuse)
- "There's no **charge** for delivery." (charge = cost/fee)

Same word, three completely different meanings. How does the model figure out which meaning applies? It looks at the surrounding words. That looking-at-other-words process is attention.

But it goes deeper than just disambiguation. Even words with clear meanings still need context:

- "She gave **him** the book." — who is "him"? Depends on earlier sentences.
- "The food was **good** but the service was **terrible**." — "good" and "terrible" relate to different nouns.
- "He couldn't **bear** to watch the **bear** attack." — two identical words, different meanings, same sentence.

Attention is the mechanism that lets each token look at all other tokens and selectively gather relevant information.

---

## The detective analogy (full version)

Imagine you're a detective at a crime scene. The room has 50 objects.

You can't spend equal time on every object. You need to figure out **what happened**, so you focus on relevant clues.

Your process:

1. **You have a question in mind**: "How did the intruder enter?"
2. **You scan every object**: broken window, muddy shoe, sleeping dog, coffee mug, open laptop...
3. **You score relevance**: broken window — very relevant. Muddy shoe — relevant. Sleeping dog — maybe (didn't bark, so intruder might be known). Coffee mug — not relevant.
4. **You spend more time on high-scoring objects**: you study the broken window closely, examine the muddy shoe prints, spend less time on the coffee mug.
5. **You build your conclusion**: "Intruder entered through the window, came from outside (mud), and the dog knew them."

Attention in a transformer does exactly this, for every token, at every layer.

---

## How attention actually works (step by step, no formulas)

Let's trace through the sentence:

**"The animal didn't cross the street because it was too tired."**

Focus on what happens when the model processes the token **"it"**.

### Step 1: Create three things for every token

For each token, the model creates three vectors (lists of numbers). Think of these as three different "cards" each token carries:

**Question card (Query):**  
What this token is looking for.  
For "it": the question card roughly encodes "I'm a pronoun — I need to find what noun I refer to."

**Label card (Key):**  
What this token advertises about itself.  
For "animal": the label card roughly encodes "I'm an animate noun."  
For "street": the label card roughly encodes "I'm a place noun."  
For "tired": the label card roughly encodes "I'm a state that applies to living things."

**Content card (Value):**  
The actual information this token carries.  
For "animal": its content includes features about being a living creature.  
For "tired": its content includes features about fatigue/exhaustion state.

### Step 2: Compare question card against every label card

The model compares "it"'s question card against the label card of every other token in the sentence.

This comparison produces a **relevance score** for each pair:

```
"it" looking at "The"      -> low score   (article, not helpful)
"it" looking at "animal"   -> high score  (animate noun, good match)
"it" looking at "didn't"   -> low score   (verb modifier)
"it" looking at "cross"    -> low score   (action verb)
"it" looking at "the"      -> low score   (article again)
"it" looking at "street"   -> medium score (noun, but place not animate)
"it" looking at "because"  -> low score   (connector)
"it" looking at "was"      -> low score   (auxiliary verb)
"it" looking at "too"      -> low score   (degree modifier)
"it" looking at "tired"    -> high score  (state implying animate subject)
```

### How does the comparison work?

Think of it like a lock and key. The question card is the lock, the label card is the key. Some keys fit the lock well (high score), some don't (low score).

The model learned during training what kinds of question-label pairs should score high. It wasn't told rules like "pronouns should attend to nouns." It discovered these patterns by reading billions of sentences.

### Step 3: Convert scores to weights

The raw scores are converted into **weights** that add up to 1 (like percentages).

```
"The"      -> 0.01  (1%)
"animal"   -> 0.40  (40%)
"didn't"   -> 0.02  (2%)
"cross"    -> 0.02  (2%)
"the"      -> 0.01  (1%)
"street"   -> 0.08  (8%)
"because"  -> 0.02  (2%)
"was"      -> 0.02  (2%)
"too"      -> 0.02  (2%)
"tired"    -> 0.40  (40%)
                     -----
Total:        1.00  (100%)
```

This is what "softmax" does — converts raw scores into a probability-like distribution. You don't need to know the math; just know it turns scores into percentages.

### Step 4: Gather information using weights

Now "it" collects information from every token, but **weighted by the scores above**.

Think of it as mixing paint:
- 40% of "animal"'s content card
- 40% of "tired"'s content card
- 8% of "street"'s content card
- Tiny amounts from everything else

The result is a **new, enriched vector for "it"** that now carries the context: "I likely refer to an animate thing that is tired."

Before attention: "it" = generic pronoun  
After attention: "it" = something like "the tired animal"

### Step 5: This happens for EVERY token simultaneously

We focused on "it" for illustration, but this exact same process happens for every token at the same time:

- "animal" also looks at all tokens and updates its own representation
- "cross" also looks at all tokens and updates its own representation
- "tired" also looks at all tokens and updates its own representation

So after one round of attention, every token in the sentence has been enriched with relevant context from the others.

---

## Why multiple heads?

The description above is one "attention head." In practice, the model runs **multiple heads in parallel** — typically 8, 12, 16, or more.

### Why? Because words have multiple types of relationships.

Take the sentence: **"The clever student who studied hard passed the difficult exam."**

Different relationships exist simultaneously:
- "student" relates to "clever" (adjective describing the student)
- "student" relates to "studied" (who studied? the student)
- "student" relates to "passed" (who passed? the student)
- "exam" relates to "difficult" (adjective for exam)
- "passed" relates to "exam" (what was passed?)

One head cannot capture all these different relationship types well. So the model uses multiple heads, each potentially learning a different type of relationship.

### Analogy: team of analysts

Imagine you're investigating a company's financial health. You send in a team:
- **Analyst A** focuses on revenue patterns
- **Analyst B** focuses on debt structure
- **Analyst C** focuses on employee trends
- **Analyst D** focuses on market competition

Each analyst looks at the same documents but asks different questions and notices different things. You combine all their findings for a complete picture.

Multiple attention heads work the same way. Each head has its own set of question/label/content cards, so each head can focus on a different type of relationship. Their results are combined into one enriched representation per token.

---

## The classroom analogy (complete version)

10 students sit in a classroom. Each student has a topic they're confused about (question card), a topic they know well (label card), and notes they can share (content card).

**One round of attention:**
1. Every student raises their question
2. Every student looks around and scores how relevant each classmate's expertise is to their question
3. Students listen more carefully to high-scoring classmates
4. Each student updates their own notes based on what they gathered

**Multiple heads:**  
This happens in parallel groups — one group discusses math relationships, another discusses science connections, another discusses historical context. All findings merge into each student's final understanding.

**Multiple layers (next files):**  
This entire classroom discussion repeats multiple times. Each round builds on the previous one, so understanding gets deeper and more refined.

---

## What attention does NOT do

This is important for building correct intuition:

- Attention does **not** change other tokens' representations directly. Each token updates only its own representation.
- Attention **does not** think or reason. It gathers and mixes information. Deep reasoning emerges from many layers of gathering + processing.
- Attention weights are **not** a perfect explanation of model decisions. They show information flow patterns, but the full story involves MLP layers and residual paths too (covered in next files).
- Attention is **not** the only important component. Without the MLP block that follows, the model would just be mixing existing information without deeply transforming it.

---

## Why attention replaced the old approach

In older models (RNN/LSTM), if you wanted to connect word 1 with word 50, the information had to travel through 49 sequential steps, getting compressed and potentially distorted at each step.

With attention, word 1 and word 50 are directly connected — one step. The model compares their question and label cards directly, regardless of distance.

This is why transformers handle long sentences so much better.

---

## Summary

| Concept | What it is | Analogy |
|---------|-----------|---------|
| Query (question card) | What this token is looking for | Detective's question at the crime scene |
| Key (label card) | What this token advertises about itself | Each object's apparent relevance |
| Value (content card) | The information this token carries | The actual clue details of each object |
| Score | How well query matches each key | Relevance rating for each object |
| Weight (softmax) | Normalized scores that add to 1 | Percentage of attention given to each object |
| Attention output | Weighted mix of all values | Detective's synthesized conclusion |
| Multiple heads | Parallel attention with different focus | Team of analysts with different specialties |

---

**Previous: [01 — Tokens and Embeddings](./01_tokens_and_embeddings.md)**  
**Next: [03 — MLP / Feed-Forward Block](./03_mlp_feed_forward.md)** — what happens to each token after attention gathers context.
