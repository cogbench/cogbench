# CogBench Constraint Audit — Full Review

## Summary

Deep audit of all 28 constraints, NLI pipeline, evaluation logic, and data quality.
7 bugs fixed so far, 6 structural issues remain.

---

## Bugs Fixed (Round 1)

| # | Bug | Impact | Fix Applied |
|---|-----|--------|-------------|
| 1 | `_find_concepts_in_text` uses substring matching — "cell" matches "cellular", "all" matches "small" | 159 false positives (22% of evals), 4 R2 pass/fail flips | Word-boundary regex with morphological suffixes (`\b{concept}(?:s|es|ed|ing)?\b`) |
| 2 | Key concepts contain stop words ("that", "which", "all") from spaCy extraction | 135 bad concepts across 120 passages inflating R2/A2/U3 | Cleaned all passage files, expanded stop_phrases in processor.py |
| 3 | 23 duplicate concepts ("cell"/"cells", "function"/"functions") counted separately | Inflating concept counts for A2/A4 | Deduplicated singular/plural across all passages |
| 4 | E4 contrastive markers include "but" and "while" → 100% pass rate for ALL models | Constraint provides zero discrimination | Removed "but" and "while", kept legitimate markers (however, although, nevertheless) |
| 5 | C3 spec markers include "with" — generic preposition | Inflates specification count | Removed "with" from marker list |
| 6 | P4 `\d+` matches any single digit — "1" triggered 49/120 times | 96.7% pass rate (nearly useless) | Require multi-digit numbers (`\d{2,}`), decimals, percentages, or specific result words |
| 7 | E2 missing "the claim"/"claims that" patterns; "should" too broad | 10/120 L5 questions failing despite clear evaluation structure | Added flexible claim patterns, removed overly broad "should" |

---

## Remaining Issues (Priority Order)

### PRIORITY 1 — U3 vs R2 Contradiction at L1 (CRITICAL)

**The problem**: U3 (passage relevance) requires ≥2 key concepts in the question. R2 (single concept) requires ≤1 key concept in the question. Both check the same `key_concepts` list against the same question text. These are nearly mutually exclusive.

**Evidence** (Phi-4 L1, 120 questions):
```
U3 + R2 cross-table:
  Both pass:          4/120   ← only these can pass L1
  U3 pass, R2 fail:  34/120  ← enough concepts for relevance, too many for Remember
  R2 pass, U3 fail:  82/120  ← single concept but fails relevance
  Both fail:          0/120
```

**Result**: L1 Remember = 0.8% to 3.3% for ALL models. Remember is supposed to be the easiest cognitive level.

**Fix options**:
- (A) Make U3 check question + answer combined (answer should reference passage too)
- (B) Lower U3 threshold to 1 for L1 specifically
- (C) Replace R2 — instead of counting concepts, check that the question doesn't use comparison/relationship language

**Recommended**: Option (A) — check passage terms in question + answer combined. This is philosophically correct: the question might be narrow ("What is mitochondria?") but the answer will reference the passage. Both together prove passage relevance.

---

### PRIORITY 2 — L3 Apply bottleneck: P3 too strict (38.5%)

**The problem**: P3 checks if the question explicitly names a method/principle from the passage. Many valid Apply questions use methods implicitly. "Calculate the force on a 5kg object accelerating at 2 m/s²" applies Newton's second law without naming it.

**Result**: L3 Apply = 4-15% across models.

**Fix options**:
- (A) Check method references in question OR answer (answer often names the method)
- (B) Add stem-based matching for methods (like we did for concepts)
- (C) Fallback to key_concepts if no methods_principles exist (already partially done)

**Recommended**: Option (A) — check both question and answer for method references.

---

### PRIORITY 3 — Only small local models (8-14B)

**The problem**: No frontier API models tested. Can't claim the benchmark evaluates "LLMs" when only testing models that fit on a single GPU.

**What's needed**: At least GPT-4o and Gemini 2.5 Flash. These are available via API keys. Together.ai can provide Llama 3.3 70B and DeepSeek-V3.

**Impact**: Without frontier models, a reviewer will question whether the benchmark discriminates at scale.

---

### PRIORITY 4 — 7/28 constraints near 100% (low discrimination)

| Constraint | Rate | Why near 100% |
|-----------|------|---------------|
| C1 (Create vocab) | 100% | Prompt instructs "create" → model uses Create verbs |
| C4 (substantial answer) | 99.8% | Create answers are naturally >50 words |
| D2 (not copied) | 99.2% | Models never copy passage verbatim |
| D4 (asks meaning) | 100% | Prompt says "explain" → always has how/why |
| E1 (Evaluate vocab) | 99.6% | Prompt says "evaluate" → uses Evaluate verbs |
| E3 (evidence request) | 99.4% | Prompt says "provide evidence" → always does |
| P2 (NLI novel scenario) | 99.2% | Generated scenarios are always novel per NLI |

**Options**:
- (A) Tighten these constraints (risk: might be measuring the wrong thing)
- (B) Acknowledge as "sanity checks" in the paper and exclude from constraint-level metric
- (C) Report "discriminative constraint rate" separately

**Recommended**: Option (B) — be transparent. Report constraint-level metric both with and without sanity checks.

---

### PRIORITY 5 — No human validation

**The problem**: The paper assumes passing all constraints = correct Bloom's level. No human study verifies this.

**What's needed**: Take 50-100 questions that pass all constraints per level. Have 2-3 annotators rate them on a 1-6 Bloom's scale. Report agreement.

**This is essential for NeurIPS**. Without it, the constraints are hypothetical proxies.

---

### PRIORITY 6 — Per-level pattern inverts Bloom's hierarchy

**Expected** (easier → harder): Remember > Understand > Apply > Analyze > Evaluate > Create

**Actual** (Phi-4): Remember(2.5%) < Apply(15%) < Analyze(52.5%) < Evaluate(56.7%) < Understand(62.5%) < Create(65%)

**Create is the easiest and Remember is the hardest.** This inverts Bloom's Taxonomy.

**Root causes**:
- L1: U3/R2 contradiction (Priority 1 fix)
- L3: P3 too strict (Priority 2 fix)
- L6: Create constraints are relatively easy (C1=100%, C3=91%, C4=99.8%)
- L2: Understand only needs how/why + paraphrasing, which models do naturally

**After fixing Priority 1 and 2**, the ordering should improve. Create being "easy" is actually defensible — LLMs are generative by nature, so producing creative output is their strength.

---

### PRIORITY 7 — Adversarial gap interpretation

**Reviewer critique**: "Adding extra constraints (wrong vocabulary) obviously reduces performance. That doesn't prove keyword reliance."

**Defense needed**: Show that the constraints REMOVED (vocabulary) account for only 1 of N constraints, so a small drop is expected, but the 12-33pp gap is far larger than losing 1 constraint would explain. The drop must be coming from the model's inability to maintain cognitive structure without the right keywords.

**Additional analysis needed**: Break down adversarial failures by constraint to show WHICH non-vocabulary constraints fail when vocabulary is swapped.

---

## Verified as Correct

- NLI model (roberta-large-mnli) — stable, no NaN, appropriate thresholds
- D3 contradiction check (< 0.50) — philosophically correct
- P2 novelty threshold (≥ 0.55) — reasonable
- C2 novelty threshold (≥ 0.60) — appropriate for creation
- Registry correctly maps constraints to levels/modes
- Adversarial mode correctly skips vocabulary constraints and adds AV/AN
- Metrics computation (strict/loose/bootstrap CI) — correct
- Extract.py parsing — robust with 4 fallback strategies
- Trigram overlap for D2 — better than single-word overlap
- Base classes (QuestionData, Constraint, ConstraintResult) — clean design

---

## Constraint Discrimination Summary (Post-Fix)

| Category | Constraints | Description |
|----------|------------|-------------|
| Strong discriminators | E2(54%), P3(39%), R1(60%), P1(60%), U3(65%) | These drive most pass/fail decisions |
| Moderate discriminators | A2(76%), R2(72%), D1(74%), P4(82%), A1(82%) | Good discrimination |
| Weak discriminators | A4(88%), A3(93%), C3(91%), E4(92%), D3(92%) | Some discrimination |
| Sanity checks | C1(100%), C4(100%), D2(99%), D4(100%), E1(100%), E3(99%), P2(99%) | Near-zero discrimination |
