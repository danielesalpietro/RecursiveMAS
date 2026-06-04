# RecursiveMAS — Extensions and Reflections
## A Supplementary Technical Document

> **Companion to:** Yang et al., *Recursive Multi-Agent Systems*, arXiv:2604.25917, 2026.
>
> This document describes engineering extensions implemented on top of the
> official RecursiveMAS release.  It is written in the spirit of a supplementary
> paper: it assumes familiarity with the base work, then adds detail on each
> extension branch, records design rationale, and ends with a reflective chapter
> on directions that go beyond benchmarks.

---

## Table of Contents

1. [Background](#1-background)
2. [HOUSE — Interactive Web Front-End](#2-house--interactive-web-front-end)
3. [LLM Enhancement Layer](#3-llm-enhancement-layer)
4. [Token Accounting and VRAM Introspection](#4-token-accounting-and-vram-introspection)
5. [Sequential-Text: Latent-Free Collaboration](#5-sequential-text-latent-free-collaboration)
6. [Multi-GPU Device Mapping](#6-multi-gpu-device-mapping)
7. [Evaluation Infrastructure Fixes](#7-evaluation-infrastructure-fixes)
8. [Reflective Chapter](#8-reflective-chapter)
9. [Summary of Branch Contributions](#9-summary-of-branch-contributions)

---

## 1. Background

### 1.1 The Core Idea of RecursiveMAS

RecursiveMAS (Yang et al., arXiv:2604.25917) introduces a framework in which
multiple heterogeneous LLM agents collaborate through *latent-space recursion*
rather than through natural-language token exchange.  Instead of passing text
between agents, the framework connects them via **RecursiveLink** modules —
compact, task-trained MLP adapters that translate hidden-state tensors across
model families, regardless of embedding dimension.  This design allows the
multi-agent loop to operate entirely in representation space, bypassing the
tokenization–detokenization bottleneck and reducing token usage by up to 75%
while achieving a 2.4× inference speedup over text-based alternatives.

The official release ships five collaboration patterns:

| Style | Topology | Agents | Total params |
|---|---|---|---|
| Sequential Light | Planner → Critic → Solver | Qwen3-1.7B · Llama3.2-1B · Qwen2.5-Math-1.5B | ~3.2 B |
| Sequential Scaled | Planner → Critic → Solver | Gemma3-4B · Llama3.2-3B · Qwen3.5-4B | ~11 B |
| Mixture | Math‖Code‖Science → Summarizer | DeepSeek-R1-Qwen-1.5B · Qwen2.5-Coder-3B · BioMistral-7B · Qwen3.5-2B | ~13.5 B |
| Distillation | Expert ⇄ Learner | Qwen3.5-9B · Qwen3.5-4B | ~13 B |
| Deliberation | Reflector → Toolcaller | Qwen3.5-4B · Qwen3.5-4B | ~8 B |

Each style is accompanied by a matching set of **Outerlinks** — the specific
RecursiveLink checkpoints that encode the inter-agent feedback paths for that
collection.

### 1.2 Scope of This Document

The extensions described here address four orthogonal concerns:

1. **Usability** — a production-grade Gradio web UI (`HOUSE`) with Chat and
   Batch Evaluation modes.
2. **Quality** — an optional LLM enhancement layer that normalises inputs and
   synthesises outputs using a frontier model (Claude, Gemini, or any
   OpenAI-compatible endpoint).
3. **Transparency** — token accounting, dynamic VRAM measurement, and a
   multi-GPU device mapper.
4. **Research** — a new *text_recursive* collaboration mode that serves as an
   ablation baseline and as an observation window into the latent communication
   channel.

All extensions live on named feature branches of the public repository and are
integrated into the development branch `claude/sharp-carson-nBKHX`.

---

## 2. HOUSE — Interactive Web Front-End

> *Branch: `feature/domain-system-prompts` (merged base) and `feature/llm-postprocessing`*  
> *File: `serve.py`*

### 2.1 Motivation

The official repository provides CLI entry points (`run.py`) that work well for
batch evaluation but create friction for ad-hoc exploration.  Researchers and
practitioners benefit from an interactive interface that allows them to ask
single questions, watch the agent pipeline run in real time, and download full
benchmark results without writing any code.

### 2.2 Architecture

`HOUSE` (*Heuristic Orchestration Using Specialist Ensembles*) is a Gradio
`Blocks` application backed by the same `run.py` / `inference_mas.py` pipeline.
Its central design principle is *warm caching*: models are loaded into VRAM on
the first request and kept alive between requests; a style switch evicts the
old models and loads the new set.  This avoids the dominant cold-start latency
(tens of seconds per model) on every interaction.

```
User browser ──► Gradio front-end
                        │
                  ┌─────┴──────────┐
                  │   serve.py      │
                  │  (warm cache)   │  ◄── _cached_load() patches
                  └──────┬─────────┘      _base.load_agent_model_and_tokenizer
                         │
                  run.py / inference_mas.py
```

The patching mechanism deserves explanation.  `inference_mas.py` calls
`load_agent_model_and_tokenizer()` at inference time — not at import time.  By
replacing that symbol on the module object *before* any inference call is made,
`serve.py` intercepts every model load with a keyed cache lookup, with no
changes to the pipeline code:

```python
_base.load_agent_model_and_tokenizer = _cached_load
_base.release_resources = _noop_release     # keep models warm
```

### 2.3 Chat Mode

The Chat tab accepts a free-form question and returns:

- **Parsed answer** extracted by `answer_utils.py`.
- **Solver output** — the full reasoning chain from Agent 3.
- **Planner / Critic outputs** in collapsible `<details>` sections.
- **Run info table** — all parameters used plus start/end timestamps and
  elapsed time, enabling exact reproducibility from the UI.

### 2.4 Batch Evaluation Mode

The Batch Evaluation tab runs a full benchmark dataset through the pipeline in
a background thread, streaming live log output to the UI.  Key design choices:

- A `threading.Lock` (`_BATCH_LOCK`) prevents concurrent batch runs.
- A `threading.Event` (`_BATCH_STOP_EVENT`) allows cooperative cancellation:
  the `_QueueWriter` stdout interceptor raises `_BatchStopped` on the next
  `write()` call after the Stop button is pressed.
- Results are written to a temporary JSONL file and offered as a download when
  the run completes.

Supported datasets: `math500`, `medqa`, `gpqa`, `mbppplus`.

### 2.5 Domain-Specific Agent Prompts

Four reasoning domains are available: `general`, `medical_emergency`,
`software_engineering`, `scientific_research`.  Each domain activates a
distinct set of role-specific system prompts:

| Domain | Planner role | Critic role | Solver role |
|---|---|---|---|
| general | General reasoning | General reasoning | General reasoning |
| medical_emergency | Trauma Senior Physician | Critical Care Specialist | Emergency Surgeon |
| software_engineering | Senior Software Architect | Senior Code Reviewer | Senior Engineer |
| scientific_research | Principal Investigator | Peer Reviewer | Science Communicator |

Domain context is propagated via `threading.local` storage so it reaches every
`render_chat_prompt()` call inside the inference pipeline without any API
change — the pipeline is entirely unaware of domain switching.

### 2.6 Versioning

The UI version is driven by a `VERSION` file at the repository root, following
semantic versioning (`major.minor.patch`).  The current release is `v1.1.1`.

---

## 3. LLM Enhancement Layer

> *Branch: `feature/llm-postprocessing`*  
> *File: `serve.py` — functions `_preprocess_question`, `_postprocess_with_llm`, `_call_llm_backend`*

### 3.1 Motivation

RecursiveMAS operates on well-formed, English-language questions.  Real users
may submit questions in other languages, with informal mathematical notation
(`x squared`), or as implicit open-ended prompts that lack the structural
clarity the pipeline benefits from.  Symmetrically, the raw MAS output is
often terse or in a format that requires interpretation.  A lightweight
pre/post-processing step using a frontier LLM addresses both ends of the
pipeline.

### 3.2 Pre-processing

The pre-processor normalises the user's question before it enters the MAS
pipeline.  It performs:

1. **Translation** to English if the input is in another language.
2. **Disambiguation** — explicit statement of implicit constraints.
3. **Structural formatting** appropriate to the target collaboration style
   (e.g., labelling domain sub-tasks for the Mixture style).
4. **Mathematical notation normalisation** (informal prose → LaTeX).

Critically, the pre-processor must *never* solve the question or add answer
options that were not already present.  The `_PRE_PROMPT_COMMON_RULES` block
encodes these prohibitions explicitly:

```
- Do NOT solve the question, compute intermediate steps, or reveal any part of the answer.
- Do NOT add multiple-choice options (A/B/C/D) if the original question does not contain them.
- Do NOT add phrases like 'Choose the correct option', 'Select one of', or similar MCQ
  instructions unless they are already present verbatim in the original question.
- Do NOT enumerate solution steps ('Step 1: ...', 'To solve this: 1. ...'). Only restate the question.
```

This rule set was strengthened after observing that Claude, when given an
open-ended calculus question, spontaneously reformatted it as a step-by-step
solution outline.  The pipeline then received a near-answer as input, causing
the solver to produce a heavily prefixed output that the answer parser could
not parse (`<NOT_FOUND>`), inflating token usage and degrading accuracy.

Multiple-choice questions (MCQ) are detected by a regex that looks for option
labels of the form `A)`, `B.`, `C)` etc. at line starts.  Detected MCQs
receive an additional prompt suffix instructing the LLM to preserve all option
text unchanged and not to hint at the correct answer.

### 3.3 Post-processing

The post-processor synthesises the MAS output (solver chain + agent
intermediates + parsed answer) into a coherent natural-language response.  It
is implemented as a separate LLM call with a structured prompt template
(`_POST_PROMPT_TEMPLATE`) that injects agent outputs up to 1200 characters
each to stay within token limits.  The post-processor also handles language
matching: if the user's original question was in Italian, the post-processor is
instructed to respond in Italian even if the MAS pipeline worked in English.

### 3.4 Supported Backends

The `_call_llm_backend()` dispatcher supports three backends:

| Label | SDK | Key source |
|---|---|---|
| Claude (Anthropic API) | `anthropic` | `ANTHROPIC_API_KEY` env var or UI field |
| Gemini (Google AI) | `google.generativeai` | `GEMINI_API_KEY` env var or UI field |
| OpenAI-compatible (vLLM / Ollama / AnythingLLM) | `openai` | endpoint URL + optional key |

API keys are **never** hardcoded in source files.  The `.env` file is listed in
`.gitignore`.  The UI field allows per-session key override without touching
the environment.

### 3.5 Model Discovery

When a backend is selected, the UI queries the provider's model list API in
real time and populates the model dropdown.  On any error (missing key, network
failure) it falls back to a hardcoded list of known model IDs.  A "Custom
model…" sentinel at the end of the list reveals a free-text input for
specifying any model ID not in the list.

---

## 4. Token Accounting and VRAM Introspection

> *Branch: `feature/llm-postprocessing`*  
> *Files: `inference_utils/_token_counter.py`, `serve.py` — `_agent_vram_gb`, `_style_vram_gb`*

### 4.1 Token Counter

Each inference module (`inference_mas.py`, `inference_mas_mixture.py`, etc.)
calls `model.generate()` internally.  Token usage is accumulated in a
thread-local counter (`_token_counter.py`) that tracks prompt tokens and
generated tokens separately.  The counter is reset at the start of each
pipeline run and emits a single `[tokens] prompt=N generated=M total=K` line
to stdout on completion.

`serve.py` parses this line with a regex and includes the counts in the Run
Info table displayed to the user.  This makes every response self-documenting:
the user can see exactly how many tokens the MAS pipeline consumed, enabling
cost estimation when the LLM enhancement layer is in use.

### 4.2 Dynamic VRAM Measurement

The original codebase used hardcoded VRAM estimates (e.g., 12 GB for
`sequential_scaled`).  When actual model sizes were measured from the HuggingFace
cache, `sequential_scaled` measured ~23 GB — nearly double the estimate.
Hardcoded values create a systematic OOM risk for users who trust the UI's
"fits / may OOM" warning.

The fix replaces static dictionaries with two functions that derive estimates
from the *same source* already used by the Model Manager tab:

```python
def _agent_vram_gb() -> Dict[str, Dict[str, float]]:
    cached = _get_cached_repos()      # reads HF scan_cache_dir()
    result = {}
    for style_name, role, repo_id in _all_model_repos():
        result.setdefault(style_name, {})[role] = cached.get(repo_id, 0) / 1024**3
    return result

def _style_vram_gb() -> Dict[str, float]:
    return {s: sum(r.values()) for s, r in _agent_vram_gb().items()}
```

For models not yet downloaded, the estimate is 0 GB (unknown), which is
conservative — the UI will not falsely warn about OOM for styles the user
hasn't downloaded.  The estimates update automatically as models are downloaded
through the Model Manager.

### 4.3 Model Manager

The Model Manager tab provides a full lifecycle interface for the 16 managed
model repositories (3–5 per style plus Outerlinks):

- **Catalog view** — shows each repo's cache status and disk size.
- **Download / Update** — triggers `snapshot_download()` in a background thread,
  streaming progress to the log.
- **Check** — queries the HuggingFace Hub for the latest commit SHA and compares
  with the local revision, flagging incomplete or stale downloads.
- **Delete** — removes cache revisions via `huggingface_hub` cache management APIs.

---

## 5. Sequential-Text: Latent-Free Collaboration

> *Branch: `feature/sequential-text`*  
> *Files: `inference_utils/inference_mas.py`, `run.py`, `load_from_repo.py`, `serve.py`*

### 5.1 Motivation and Relation to the Paper

The official RecursiveMAS paper (arXiv:2604.25917) includes a **text-recursive**
ablation: the same Planner → Critic → Solver topology, but with agents
communicating via natural-language tokens instead of latent tensors.  This
baseline was already fully implemented inside `inference_mas.py` as the
`text_recursive` code path, but was never exposed in the CLI or the web UI —
the `args.method` variable was hardcoded to `"ours_recursive"`.

Exposing this path serves two purposes:

1. **Research reproducibility** — users can now run the paper's ablation from
   the same CLI and UI as the primary method.
2. **Generalisation** — without RecursiveLinks, any instruction-tuned model
   pair can be plugged in, making the text mode useful for rapid prototyping
   with arbitrary models from HuggingFace.

### 5.2 Implementation

Three changes were required:

**`inference_mas.py`** — The hardcoded `args.method = "ours_recursive"` was
replaced with a CLI argument:

```bash
--method {ours_recursive,text_recursive}   default: ours_recursive
```

All adapter validation (inner aligner paths, outer path resolution) was made
conditional on `_is_latent_method = args.method == "ours_recursive"`.  This
allows the text mode to bypass all RecursiveLink requirements without touching
the latent-mode code paths.

**`run.py`** — The `resolve_style_paths()` function gained a `text_sequential`
family branch that returns only the three agent model paths (no Outerlinks, no
adapters).  The `build_cli_for_style()` dispatcher passes `--method
text_recursive` to `inference_mas` when the family is `text_sequential`.

**`load_from_repo.py`** — A new `sequential_text` entry was added to
`STYLE_SPECS`:

```python
"sequential_text": {
    "family": "text_sequential",
    "repos": {
        "planner": "RecursiveMAS/Sequential-Light-Planner-Qwen3-1.7B",
        "critic":  "RecursiveMAS/Sequential-Light-Critic-Llama3.2-1B",
        "solver":  "RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B",
    },
},
```

The default models are the Sequential Light checkpoints, keeping memory
requirements low (~9 GB for three models without Outerlinks).

### 5.3 Model Selection UI

In the `sequential_text` style, users can override each agent's model
independently.  The UI presents:

- A **dropdown** listing all LLM models currently in the local HF cache
  (Outerlinks excluded, sorted by size descending), plus a "Custom model…"
  sentinel.
- A **free-text field** (shown only when "Custom model…" is selected) for
  entering any HuggingFace model ID.

This design avoids exposing an infinite public-repo search while still allowing
the user to specify any model, downloaded or not (the serving code will trigger
`snapshot_download` on first use).

The three override dropdowns are grouped in a collapsible `gr.Accordion`
(`🔧 Model Configuration`) that is only visible when `sequential_text` is
selected, keeping the UI clean for the standard latent styles.

### 5.4 Semantic Significance

The `sequential_text` mode is not merely an engineering convenience.  It is the
most natural entry point for *observing what the agents actually think*.  In
the latent-recursive styles, inter-agent communication happens in embedding
space — the intermediate representations are not human-readable.  In
`sequential_text`, every message between agents is a natural-language string
that can be logged, inspected, and analysed.

This makes the text mode the primary debugging and interpretability interface
for the latent-recursive system: run the same question in both modes, compare
the intermediate outputs, and observe which information the latent path encodes
compactly but opaquely, versus what the text path makes explicit.  The text
mode also generates natural supervision signal for fine-tuning new specialist
models, as discussed in Section 8.

---

## 6. Multi-GPU Device Mapping

> *Branch: `feature/multi-gpu`*  
> *File: `serve.py` — functions `_mg_update_style`, `_mg_toggle_enabled`, `_mg_apply`*

### 6.1 Motivation

The five collaboration styles span a wide range of memory requirements (5 GB
for Sequential Light up to 23 GB for Sequential Scaled).  Consumer hardware
commonly includes two GPUs (e.g., a primary RTX 4090 at 24 GB and a secondary
RTX 3090 at 24 GB).  The `accelerate` library and `device_map` arguments in
Transformers allow individual model components to be placed on specific devices,
but the RecursiveMAS pipeline does not expose this to the user.

### 6.2 Design

The Multi-GPU tab allows users to assign each agent in a style to a specific
GPU.  The assignment is stored in a `gr.State` dictionary keyed by style name:

```python
device_map_state: Dict[str, Dict[str, str]]
# e.g. {"sequential_scaled": {"planner": "cuda:0", "critic": "cuda:0", "solver": "cuda:1"}}
```

This state is passed through the `respond()` and `run_batch_eval()` call
chains and forwarded to `build_cli_for_style()` which injects it as
`--device_map` CLI arguments to the relevant inference module.

The UI shows a per-agent VRAM check when the user applies an assignment:

```
🟢 Planner on cuda:0: ~8.2 GB — OK
🟡 Solver on cuda:1: ~7.4 GB — OK
```

The check uses `_agent_vram_gb()` (see Section 4.2) for size estimates and
`torch.cuda.mem_get_info()` for live free-VRAM readings.

### 6.3 The `min_width` Fix

An earlier version of the multi-GPU UI used `gr.Row()` with fixed column
widths.  On narrow viewports (tablet, windowed desktop), Gradio's layout engine
would collapse columns below their minimum width and render overlapping widgets.
The fix removed explicit `min_width` constraints and relied on Gradio's
responsive flexbox layout instead.

---

## 7. Evaluation Infrastructure Fixes

> *Branch: `feature/llm-postprocessing`*  
> *Files: `inference_utils/answer_utils.py`, `inference_utils/inference_mas.py`*

Three systematic evaluation bugs were identified during Math500 benchmarking
and fixed:

### 7.1 `compare_answers()` Cascading A-Default

The original call site passed `default="A"` for both the predicted and gold
answer arguments.  When the answer parser failed to extract an answer from
either side, it silently substituted `"A"`.  For questions with gold answer
`"A"`, a parse failure on the prediction side was silently scored as *correct*.
This inflated accuracy by a systematic amount equal to the failure rate
multiplied by the fraction of questions with gold `"A"`.

**Fix:** `default=None` on both sides; `correct=False` whenever either side
cannot be parsed.

### 7.2 `extract_gold_answer()` Silent A-Default

Similarly, unparseable gold answers for choice-format datasets silently became
`"A"`, skewing the gold-answer distribution and masking dataset quality issues.

**Fix:** Return `None` on failure; propagate the `None` to `compare_answers()`
where it is handled by the fix above.

### 7.3 Latent Solver Output Discriminator

The `run_solver_latent_stage()` function needed to distinguish between
generation-only output and prompt-prefixed output.  The original heuristic
compared the output length with `max_new_tokens` — unreliable when the model
generates a very short answer (e.g., `\boxed{B}`, which is 9 tokens, well
below any typical `max_new_tokens` value).

**Fix:** Use `prompt_len` (the actual token length of the prompt tensor) as the
discriminator.  The output is generation-only iff `len(output) == generated_len`
with no prompt prefix.

### 7.4 Dataset Gold-Answer Errors

A specific Math500 question about a definite integral was found to have a
gold answer of `A` (= 1) when the mathematically correct answer is `B` (= 2).
The MAS correctly computed the answer as 2 but was scored as `correct=False`.
This is a dataset error rather than a model or pipeline bug.  It is documented
here as a reminder that benchmark accuracy numbers carry inherent noise from
annotation errors that cannot be corrected at inference time.

---

## 8. Reflective Chapter

This chapter steps back from engineering specifics to consider broader
questions raised by RecursiveMAS.  It is explicitly speculative in parts; the
intent is to frame research questions rather than present results.

### 8.1 What the Latent Channel Actually Encodes

In the standard RecursiveMAS modes, agents exchange hidden-state tensors
rather than text.  This raises a question that is surprisingly difficult to
answer: *what information is carried by the latent channel, and how does it
differ from what text exchange would carry?*

The most direct approach is the comparative experiment enabled by the
`sequential_text` mode introduced in Section 5.  By running the same questions
in both modes under identical model weights, and examining:

- the token count difference (latent is dramatically lower);
- the intermediate text outputs in the text mode versus post-hoc probing of
  latent representations;
- accuracy differences as a function of question type;

one can begin to characterise what is *lost* in the latent compression.  Anecdotal
evidence from the debugging process suggests the latent channel is
particularly good at transmitting structural/relational information (what
operations to apply, in what order) while losing surface-level justifications
(the natural-language reasoning steps that would appear in a chain-of-thought).

This has a practical implication for alignment: a system that communicates
primarily in latent space is harder to audit than one that communicates in text.
The text mode is therefore not merely an ablation baseline — it is the
*interpretability interface* for the latent system.

### 8.2 Vertical Specialisation

The Mixture style already embeds a notion of domain specialisation: Math, Code,
and Science agents process the same question from their respective angles, and
the Summarizer aggregates the results.  A natural extension is to train more
specialised vertical models — not just domain specialists but *role* specialists:

- A **Medical Diagnostician** trained exclusively on USMLE, MedQA, and
  clinical case reports.
- A **Formal Verifier** fine-tuned on theorem-proving corpora (Lean, Coq, Isabelle).
- A **Quantitative Analyst** trained on financial reasoning and numerical
  methods.

The RecursiveLink mechanism would then need to accommodate a much larger matrix
of adapter pairs.  With $N$ specialists, $N(N-1)$ directional adapters are
needed for full connectivity — growing quadratically.  Sparse topologies
(each specialist connects only to the Summarizer and possibly one or two
related specialists) reduce this to $O(N)$.

An important boundary condition: the RecursiveLink between two agents is only
as good as the training distribution used to learn it.  A link trained on Math500
questions will not reliably transfer information from a Medical agent to a
Legal agent.  This motivates research into *universal adapters* — perhaps
trained on broad multi-domain datasets — as a default fallback for pairings not
explicitly trained.

### 8.3 Constitutional Constraints on Recursive Reasoning

A multi-agent system that iterates over its own outputs introduces risks that
are qualitatively different from those of a single-pass model:

**Echo chamber / mode collapse.** If all agents were fine-tuned on similar
data with similar objective functions, the recursive loop may amplify a shared
bias rather than correct it.  The Critic agent in particular may have learnt to
validate rather than challenge the Planner — especially if the training
distribution rewarded agreement.  Evidence: accuracy improves modestly over
rounds in the paper's experiments, suggesting the recursive update does provide
genuine signal, but the gain diminishes, which is consistent with partial echo
chambers.

**Galaxy-brained reasoning.** Iterative self-consistent reasoning can lead
an agent to commit to an internally coherent but factually wrong chain of
reasoning.  Each recursion round reinforces the prior round's conclusion.
The Planner–Critic interaction is designed to counteract this, but only if the
Critic has genuinely orthogonal knowledge or a different inductive bias.

**Oscillation.** In theory, the recursive update could oscillate without
converging.  The paper reports convergence in practice, but this has not been
studied rigorously as a function of model diversity, learning rate of the
adapters, or question difficulty.

These failure modes motivate *constitutional constraints* on the recursive
loop — rules, enforced either at training time or inference time, that prevent
specific failure patterns:

- A **diversity constraint** on the Critic: it must produce an output whose
  semantic embedding differs from the Planner's output by at least a minimum
  cosine distance.
- An **uncertainty budget**: after $k$ rounds with no change in the parsed
  answer, the loop terminates rather than consuming further computation.
- An **external verifier** for high-stakes domains: before accepting the Solver's
  output, route it through a domain-specific validator (a formal checker, a
  unit-test suite, a medical guideline lookup) and feed the verdict back as
  a hard constraint.

These ideas are analogous to Constitutional AI (Bai et al., 2022), which uses
a fixed set of written principles to guide an LLM's self-revision process.
The key difference is that Constitutional AI operates at the single-model
level via text prompts, whereas the recursive MAS architecture could embed the
constraints directly into the adapter training objective.

### 8.4 Self-Improvement via Text-Recursive Data Collection

The `sequential_text` mode generates a corpus of labelled trajectories: for
each question, the complete sequence of Planner output → Critic output →
Solver output is available as text, along with the final correctness label.

This creates a natural data flywheel:

```
Run sequential_text on benchmark
       ↓
Filter correct=True trajectories
       ↓
Fine-tune new specialist models on filtered data
       ↓
Evaluate new models on held-out benchmark
       ↓
Repeat
```

This is structurally similar to STaR (Self-Taught Reasoner, Zeiler et al., 2022)
and to the DeepSeek-R1 self-play loop.  The key difference in the RecursiveMAS
setting is that the trajectory is *multi-agent*: the Planner's output that led
to a correct Solver output is automatically identified as a good planning
strategy for that class of question, and similarly for the Critic's intervention.

An important subtlety: filtering on `correct=True` introduces a selection bias
towards easier questions (or questions well-matched to the current models).
A curriculum strategy — starting with easy questions and progressively
increasing difficulty — may be necessary to prevent the fine-tuned models
from becoming overconfident on easy instances while failing on hard ones.

The text-recursive corpus also provides *supervision signal for the RecursiveLink
adapters themselves*: if a text trajectory is correct, the latent path should
ideally carry the same information.  One could train a contrastive objective
that encourages the latent representation to preserve the semantic content of
the corresponding text, modulated by a term that rewards compression (fewer
bits per message).

### 8.5 The Deliberation Style and External Tool Integration

The Deliberation style (Reflector → Toolcaller) is the most architecturally
distinct of the five patterns because it includes an *actuator*: the Toolcaller
agent can invoke real-world tools (web search via Tavily, Python code execution).
This makes the recursive loop an agent in the full sense — a system that
observes, reasons, and acts.

Current implementation uses a fixed tool list defined at launch time.
A natural extension is a *tool discovery* protocol: the Toolcaller dynamically
selects which tools to call based on a learned embedding-space matching between
the problem latent and a catalogue of tool signatures.

A harder question is *tool trust*: if the web search returns a wrong answer
(or, adversarially, a planted wrong answer), the Toolcaller will incorporate
it into its latent state and pass a corrupted signal to the Reflector.
Constitutional constraints could address this: the Reflector's cross-checking
function should specifically verify claims that came from external tool calls,
applying a higher scepticism weight.

### 8.6 Latent Space as a Private Communication Channel

There is an anthropomorphic temptation to describe the latent-space exchange
as agents "thinking together" — sharing pre-linguistic thoughts that are
richer and faster than any text they could articulate.  This framing is
probably wrong in detail (the representations are not thoughts in any
philosophically loaded sense) but useful as a design intuition.

What the framing correctly captures is the *privacy* property: the latent
messages are not readable by a human observer in real time.  Unlike a
chain-of-thought that can be logged and audited, the latent channel is opaque.
As systems like this are deployed in higher-stakes settings, the interpretability
gap between the text mode (fully auditable) and the latent mode (partially
auditable via probing) will need to be closed.

One direction: train a *translator network* that maps latent messages to
natural-language summaries on demand.  This would not slow down the inference
loop (translation happens post-hoc), but would allow auditors to inspect what
any agent "said" to any other agent on any specific inference step.

---

## 9. Summary of Branch Contributions

| Branch | Key files modified | New capability |
|---|---|---|
| `feature/domain-system-prompts` | `prompts.py`, `inference_mas.py`, `serve.py` | Domain-specific system prompts; HOUSE two-tab layout; batch evaluation; run metadata; advanced settings; warm model cache |
| `feature/llm-postprocessing` | `serve.py`, `inference_utils/answer_utils.py` | LLM pre/post-processing layer; MCQ detection; dynamic VRAM from HF cache; token counting; 3 evaluation bug fixes; Model Manager check/update |
| `feature/sequential-text` | `inference_mas.py`, `run.py`, `load_from_repo.py`, `serve.py` | `sequential_text` style; CLI `--method text_recursive`; model selection UI |
| `feature/multi-gpu` | `serve.py` | Per-agent GPU assignment UI; live VRAM fitness check; `min_width` layout fix |

All branches are pushed to `origin` and the development branch
`claude/sharp-carson-nBKHX` integrates changes for validation.

---

## References

Yang, X., Zou, J., Pan, R., Qiu, R., Lu, P., Diao, S., Jiang, J., Tong, H.,
Zhang, T., Buehler, M. J., He, J., & Zou, J. (2026). *Recursive Multi-Agent
Systems*. arXiv:2604.25917.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., … &
Kaplan, J. (2022). *Constitutional AI: Harmlessness from AI Feedback*.
arXiv:2212.08073.

Zeiler, M., Kaddour, J., & others. (2022). *STaR: Self-Taught Reasoner —
Bootstrapping Reasoning With Reasoning*. NeurIPS 2022.

DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
via Reinforcement Learning*. arXiv:2501.12948.

---

*Document version: 1.0.0 — 2026-06-04*
