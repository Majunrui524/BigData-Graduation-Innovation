<div align="center">

# ⚡ Big Data Graduation Innovation

**Unsupervised Social Bot Detection with Structural-Entropy Community Detection on Multi-Relational Graphs**

<p align="center">
  <img src="docs/screenshots/02-graph.png" alt="Encoding-tree community graph: 898 communities on 18,743 users" width="92%">
</p>

Detect coordinated social-bot networks **without a single training label**. LLM-assisted multi-view feature fusion → adaptive late-fusion graph construction → structural-entropy encoding-tree community discovery → purity-based interpretation.

**[🕵️ Try the Account Detective →](https://Majunrui524.github.io/BigData-Graduation-Innovation/#/detective)** · [🚀 Live Demo](https://Majunrui524.github.io/BigData-Graduation-Innovation/) · [🧠 Research Highlights](#-research-highlights) · [⭐ Star](https://github.com/Majunrui524/BigData-Graduation-Innovation)

</div>

---

## 🎯 TL;DR — One Glance, One Minute

> **The problem.** Twitter/X is full of coordinated bot armies that spread spam, inflate
> trends, and manipulate public opinion. Today's detectors have three fatal weaknesses:
> they need **thousands of human-labeled accounts** to train, they break the moment bots
> change their tricks, and they tell you "bot / human" but never *why*.
>
> **The flip.** This project detects bot networks with **zero labels**. It turns each
> account into four evidence views (what they post, how they post, when they post, whom
> they follow), fuses them into one weighted user graph, and lets suspicious communities
> emerge by **minimizing the structural entropy** of that graph. No training, no
> annotation, no GPU. Labels are used *only* afterwards to check the result — never to
> guide the search.
>
> **The result.** On the TwiBot-22 benchmark (18,743 accounts sampled), it discovers
> **898 communities** with the **lowest structural entropy** (12.3537 vs 15.99 for
> K-Means), the **most compact bot clusters** (largest community: 283 vs 2,734 users),
> and a post-hoc label purity of **0.8643** — a structural map, not just a verdict.
>
> **What you can do with it.** Click any account in the
> [interactive demo](https://Majunrui524.github.io/BigData-Graduation-Innovation/#/detective)
> and read its full bot/human evidence chain. Or reproduce every headline number on your
> own machine in 5 seconds:
>
> ```bash
> python project-code-implementation/tools/offline_reproduce.py
> ```

---

## ⚡ Reproduce in 5 Minutes — Data, Train, Compare

> **"Reproducible" is the #1 reason researchers star a project.** Here is the full recipe,
> from raw data to the exact numbers on this page.

### Level 0 · Sanity check — 5 seconds, zero setup

```bash
git clone https://github.com/Majunrui524/BigData-Graduation-Innovation.git
cd BigData-Graduation-Innovation
python project-code-implementation/tools/offline_reproduce.py
```

This single command reloads the shipped per-user + per-community JSON bundle and
**recomputes all 11 headline numbers** (18,743 users · 898 communities · 0.8643 purity ·
largest 283 · median 13 · …). No install, no API key, no GPU. **Exit code 0 = every number
on this README is verified.** This is the fastest way to confirm the results are real.

### Level 1 · Full pipeline — data download → training → comparison

```bash
cd project-code-implementation

# 1. Environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .                                        # numpy, scikit-learn, gensim, ijson (no GPU)

# 2. Download the official TwiBot-22 benchmark (~several GB, academic license)
bash scripts/download_twibot22.sh --tiny                # preview the recipe first
bash scripts/download_twibot22.sh                       # gdown from the authors' Google Drive

# 3. Sample a 10k-user subset (reproduces every number on this README; seed 42 is pinned)
python -m twibot22_sampler.cli sample --preset main \
  --data-root   data/twibot22_raw \
  --work-root   data/work_main \
  --output-root data/samples/final_v1 \
  --seed 42

# 4. LLM-assisted semantic extraction (post-type + triplet compression; needs an API key)
cp .env.example .env                                    # fill in your OpenAI-compatible key
bash scripts/run_full_api_pipeline.sh

# 5. Embeddings → late-fusion graph → structural-entropy communities → evaluation
bash scripts/run_10k_late_results.sh
```

**Compare.** Your run writes its output under
`data/samples/final_v1/analysis/run_10k_late/` (`graph_manifest.json`,
`community_manifest.json`, `community_eval_manifest.json`). The table below is exactly
what you should reproduce:

| Method | Communities | Largest | H(P) ↓ | Modularity | Density | Clustering | Global Purity |
|---|---|---|---|---|---|---|---|
| K-Means | 898 | 2,734 | 15.9861 | 0.1959 | 0.0030 | 0.0646 | 0.8625 |
| Late Fusion + Weighted LPA | 241 | 5,798 | 14.2002 | **0.6439** | 0.0575 | 0.1923 | 0.8641 |
| **Late Fusion + Structural Entropy (Ours)** | 898 | **283** | **12.3537** | 0.5130 | **0.1650** | **0.3366** | **0.8643** |

Want a fast end-to-end smoke test first? Use `--preset smoke` (2,000 users) — the same
chain, ~10× faster, same code path.

---

## ❓ Why This Project — 项目背景

**The 60-second version.** Imagine a factory that rents 10,000 fake Twitter accounts.
Those accounts cannot just post — to look real, they must **follow each other, retweet
each other, and stay active around the clock**. That coordination is the one thing a bot
cannot fake, because it is *structural*: it lives in the shape of the network, not in the
text of any single tweet. This project hunts that structure instead of chasing the words.

**Why existing detectors are losing the arms race:**

| Weakness | What it means | Why it fails today |
|---|---|---|
| **Label-hungry** | Supervised models need thousands of human-annotated accounts (expensive, slow, stale) | Bot operators rotate accounts; labels age in weeks |
| **Feature-blind** | Lexical/style/statistical features are static fingerprints | GPT-4-class bots rewrite style on demand — features stop working the day they're published |
| **Black-box** | You get "0.93 bot" with no evidence chain | Researchers and platforms need to *show why* an account is flagged |

**The paradigm flip.** Stop optimizing for what a bot *says*, and start reading what a bot
*does to its neighborhood*. Coordinated bot rings must form dense mutual-following
clusters to look legitimate — so the network itself is the signal. Structural-entropy
minimization finds those clusters with no labels at all, and the encoding tree exposes the
evidence at every scale.

---

## 🧭 Design at a Glance — 整体设计思路

**One sentence:** *Turn every account into four evidence views, fuse them into one
weighted graph, then let anomalous communities emerge from the graph structure.*

**Five stages, plain language:**

```
  ① COMPRESS ──▶ ② FUSE ──▶ ③ GRAPH ──▶ ④ DISCOVER ──▶ ⑤ INTERPRET
```

| # | Stage | Plain language | What it produces |
|---|---|---|---|
| ① | Multi-view compression | Squeeze each account into 4 complementary angles: *what* they post (LLM semantic triplets + post-type), *how* they post (frequency, URL/JS diversity), *when* they post (24-h activity rhythm), *whom* they follow (neighborhood overlap) | 4 feature views per user |
| ② | Adaptive late fusion | Blend the 4 views into one similarity score — but re-normalize **only over the views actually observed**, so accounts with missing data are never unfairly punished | 1 similarity matrix per view + fused weights |
| ③ | Weighted graph | Users are nodes; fused similarity is edge weight | 1 weighted user graph |
| ④ | Structural-entropy discovery | Greedily merge user groups that most reduce the encoding cost of the graph — small bot rings and large human regions surface at their natural scale | 898 communities (18,743 users) |
| ⑤ | Post-hoc interpretation | *After* discovery, check each community against labels to name it: pure-human macro-region, compact bot cluster, mixed transitional zone, sparse periphery | Archetype map + purity 0.8643 |

**Why four views and not one?** No single view is enough. A human can tweet at bot-like
frequency during a PR crisis; a bot can imitate human posting rhythm. But the four views
are *weakly correlated* — an account that looks human in content **and** behaves like a
machine **and** sits inside a dense mutual-following ring is almost certainly coordinated.
Their disagreement is information, not noise.

**The core principle, in one line:** *labels are used to check the result, never to build
it.* The optimization objective is purely structural (entropy ↓); the reported purity is a
validity indicator only.

> Deep dive (for the technical reader): the full method — equations, per-view features,
> the encoding-tree formulation — is in [Research Highlights](#-research-highlights)
> below.

---

## ❔ Zero-to-One FAQ — 细节解答

**Q1. "Unsupervised" — what does that actually mean here?**
It means the algorithm never sees a bot/human label while finding communities. It only
reads the graph structure (who is connected to whom, how strongly). Labels enter the
pipeline **after** discovery, purely to name what was found. That is why the method cannot
"overfit to old labels" — there are no labels to overfit to.

**Q2. What is structural entropy? Give me a picture.**
Think of moving into a new apartment with 10,000 boxes. You could leave them in one giant
pile (one "community": fastest to put down, painful to use) or split them into 10,000
single boxes (perfectly organized, absurd to manage). Structural entropy measures the
*cost of describing the whole pile with a tree*. The algorithm greedily merges groups
that most lower that cost, and stops at the natural scale — like finding that 898 boxes
is the sweet spot where everything is findable without over-engineering.

**Q3. What does 0.8643 purity mean?**
For each discovered community, count which label (bot or human) is the majority, then
average that majority share over all communities, weighted by size:
`1/|C| · Σ max_y P(y|C) = 0.8643`. It says: *if we had trusted the discovered structure
alone, we would have agreed with the human annotators on 86% of accounts* — and we never
used their labels to find it.

**Q4. Do I need a GPU? An API key?**
Neither for the core method — the community-discovery and evaluation stages are pure
NumPy/sci-kit-learn (CPU, laptop-friendly). An **OpenAI-compatible API key** is only
needed for the optional LLM stage (semantic triplets + post-type compression). The
zero-dependency `offline_reproduce.py` needs nothing but Python 3.8+.

**Q5. Where is the dataset, and how big?**
The raw TwiBot-22 corpus is ~1M users / 170M relations (several GB) and is **not**
committed to this repo. Download it with `scripts/download_twibot22.sh` (official Google
Drive mirror, CC BY-NC-ND 4.0, academic use only). The demo already ships with the
precomputed 10k analysis bundles, so it works out of the box.

**Q6. How does this compare to supervised methods like BotRGCN?**
They solve different problems. BotRGCN-type models can reach ~0.80 accuracy **given
labeled training data** for the same distribution. This project needs **zero labels**,
generalizes across bot generations, and returns an interpretable community map instead of
a scalar score. In the comparison table above, it beats both unsupervised baselines
(K-Means, Weighted LPA) on entropy, cohesion, and purity — with labels used only
post-hoc.

**Q7. The demo says "10k" but the README says 18,743 — which is it?**
Both. The pipeline sampled a **10,000-user core** and then expanded with graph context to
**18,743 accounts** that have at least one view of evidence. Every number on this page is
computed on the full 18,743-account bundle shipped in `demo/public/data/10k/`.

**Q8. Can someone else reproduce my exact experiment?**
Yes — the random seed is pinned (`--seed 42`), the sample presets are fixed
(`smoke` = 2,000 users, `main` = 10,000), and the entire evaluation pipeline is
deterministic given the same inputs. The 26-module pytest suite locks the behavior.

**Q9. Can I run this on my own social-graph data?**
The four-view pipeline is data-format agnostic at the graph stage. Provide your own
user/edge tables in the same layout (see `src/twibot22_sampler/readers.py`) and re-run
stages ③–⑤ (`build-user-graph` → `detect-communities` → `evaluate-communities`). No LLM
stage is required if you build the views yourself.

**Q10. Why is the repository fully anonymous?**
To keep the research self-contained and citation-neutral: commits are authored as
"Anonymous", no personal/affiliation info appears anywhere, and the thesis PDF is not
included (the full method is documented in this README instead).

---

## 🗂️ Repository Structure

```
bigdata-graduation-innovation/
├── project-code-implementation/         # 🐍 Main implementation
│   ├── src/twibot22_sampler/            # Core Python package (~12k lines)
│   │   ├── cli.py                       #   CLI entry: 10+ subcommands
│   │   ├── user_graph.py                #   Multi-view graph construction (late fusion)
│   │   ├── structural_entropy.py        #   Structural-entropy community detection
│   │   ├── llm_client.py                #   Zero-dependency OpenAI-compatible client
│   │   ├── user_features.py             #   Behavior / profile feature engineering
│   │   ├── temporal_profiles.py         #   Circadian rhythm + DTW modeling
│   │   ├── triplets.py / post_types.py  #   LLM semantic compression
│   │   └── community_*.py               #   Evaluation, purity, reranking, analysis
│   ├── tests/                           # 26 pytest modules
│   ├── tools/                           # Frontend data-bundle generators
│   ├── demo/                            # ⚛️ React + Vite interactive dashboard
│   ├── scripts/                         # End-to-end run + data-download scripts
│   ├── pyproject.toml
│   └── .env.example                     # API settings template (never commit .env)
└── .github/workflows/                   # GitHub Actions (demo → Pages)
```

---

## 📊 Key Results (10k sampled subgraph)

The encoding-tree partition achieves the **lowest structural entropy**, the **most compact
largest community** (283 vs 2,734 users), the **strongest local cohesion** (density ×55,
clustering ×5 over K-Means), and the **highest label-aware purity** — all without
supervised training.

**Community archetypes discovered (post-hoc):** 118 pure-human macro-communities ·
103 compact bot communities · 215 mixed transitional communities · 462 sparse peripheral
fragments.

---

## 🧠 Core Idea

```
                    ┌────────────────────────────────────────────────────────────┐
                    │                     TwiBot-22 (sampled)                   │
                    │              18,743 users · 202,556 tweets               │
                    └───────────────────────────┬────────────────────────────────┘
                                                │
        ┌───────────────────────────────────────┼───────────────────────────────────────┐
        │            LLM-assisted multi-view feature extraction                       │
        ▼            ▼                            ▼                            ▼
   ┌──────────┐ ┌──────────┐              ┌──────────────┐              ┌──────────┐
   │ Content  │ │ Behavior │              │   Temporal   │              │ Network  │
   │triplets +│ │statistics│              │  circadian   │              │following │
   │post types│ │+ JS div. │              │   DTW hist.  │              │Jaccard + │
   │  (LLM)   │ │          │              │              │              │  degree  │
   └────┬─────┘ └────┬─────┘              └──────┬───────┘              └────┬─────┘
        └────────────┴──────────────────────────┴────────────────────────────┘
                                                │
                                    Adaptive late fusion (Eq. 3.13)
                                                │
                                          ┌─────▼─────┐
                                          │ Weighted  │   missing modalities are
                                          │ user graph│   re-normalized, never punished
                                          └─────┬─────┘
                                                │
                                   Structural-entropy minimization
                                   (greedy agglomerative encoding tree)
                                                │
                                    ┌───────────▼───────────┐
                                    │  898 communities with  │
                                    │ 4 archetypes (pure bot,│
                                    │ pure human, mixed, ...)│
                                    └───────────┬───────────┘
                                                │
                                    Post-hoc purity evaluation
                                    (labels used only to interpret, never to optimize)
```

<p align="center">
  <img src="docs/figures/poster-01-bots-comparison.png" alt="Rule-based traditional bots vs. LLM-driven modern AI bots — five dimensions where old detectors are losing the arms race" style="max-width: 1100px; width: 100%; display: block; margin: 1.5rem auto;">
</p>

---

## 🧠 Research Highlights

> This repository presents an **unsupervised social bot detection framework** — a zero-label, graph-centric pipeline for discovering coordinated bot networks. The core innovations are summarized below.

### Method Overview

The pipeline proceeds in five tightly-coupled stages (Fig. 1). Each TwiBot-22 user is first compressed into a four-channel feature vector spanning *what* they post (LLM triplets + post-type), *how* they post (statistics + URL/JS diversity), *when* they post (24-hour DTW-aligned activity), and *whom* they follow (Jaccard similarity over their ego-graph). The four channels are then fused into a single weighted user graph via an **adaptive late-fusion** scheme that re-normalizes view weights over the set $\mathcal{O}$ of observed modalities only, so accounts with missing data are not unfairly penalized. A **structural-entropy encoding tree** is then built greedily over this graph: at every step we merge the pair of communities whose union most reduces structural entropy $\mathcal{H}(\mathcal{T})$, and the entropy minimum identifies the natural community scale. The 898 resulting communities are grouped into four structural archetypes — pure-human macro-regions, compact bot clusters, mixed transitional zones, and sparse periphery — and characterized *post-hoc* by a label-aware purity score. **Crucially, ground-truth labels never enter the optimization; they only serve to interpret an already-discovered structure.**

### Why Unsupervised, Why Graph-Centric

Modern AI-driven bots (GPT-3 / LLaMA / GPT-4-class) generate context-aware, human-indistinguishable content in real time, breaking every content-feature detector that relied on lexical or stylistic anomalies. Rule-based and supervised detectors are losing the arms race: they depend on static features, expensive labels, and cannot generalize to zero-day bot strategies. By using the **graph structure itself** as the signal — coordinated bot networks *must* densely connect to one another to appear legitimate — the approach is robust to content obfuscation and requires no labeled training data at all. The four views in Fig. 2 give four independent angles on the same account; their disagreement is informative rather than fatal, because the late-fusion step (Fig. 3) re-normalizes over whichever views are actually observed.

### Four Complementary Evidence Views

<p align="center">
  <img src="docs/figures/poster-02-content-view.png" alt="Content view — LLM semantic compression: raw tweets → LLM → normalized embedding + (subject, predicate, object) triplets → 1536-d user vector" style="max-width: 1100px; width: 100%; display: block; margin: 1.5rem auto;">
  <img src="docs/figures/poster-03-temporal-view.png" alt="Temporal view — 24-hour activity distribution: humans show diurnal peak, bots are nearly uniform (DTW distance)" style="max-width: 1100px; width: 100%; display: block; margin: 1.5rem auto;">
  <img src="docs/figures/poster-04-network-view.png" alt="Network view — diffusion topology of fake vs. normal users: bots form dense mutual-following clusters" style="max-width: 1100px; width: 100%; display: block; margin: 1.5rem auto;">
</p>

Each view captures a different facet of the human/bot distinction, and *no single view is sufficient*. **(a) Content** uses an LLM triplet-encoder to compress raw post text into a compact embedding, with post-type frequency acting as a coarse category prior. **(b) Behavior** summarizes posting volume, JS-string diversity, URL share, and retweet ratio — bot accounts cluster on extreme values (very high frequency, near-zero JS diversity). **(c) Temporal** measures the 24-hour activity curve via Dynamic Time Warping against a canonical human diurnal pattern; bot accounts are nearly uniform across the day. **(d) Network** uses Jaccard similarity over followee sets to surface dense mutual-following clusters, the canonical signature of bot rings. The four views are weakly correlated, so a single account can be confidently classified only by *fusing* them (Fig. 3).

### Adaptive Late-Fusion Graph Construction

<p align="center">
  <img src="docs/figures/poster-05-adaptive-fusion.png" alt="Adaptive late-fusion: per-account normalization over observed modalities, missing views are gracefully re-weighted instead of imputed" style="max-width: 1100px; width: 100%; display: block; margin: 1.5rem auto;">
</p>

Naive concatenation penalizes accounts with missing modalities (e.g., a private account with no public posts); naive averaging lets a single noisy modality dominate. Our adaptive scheme computes one similarity matrix per view, then forms a fused edge weight by re-normalizing each view's base importance $\hat{w}_i$ **only over the views that are observed** for the account pair under consideration. This gives $w_i = \hat{w}_i \big/ \sum_{j \in \mathcal{O}} \hat{w}_j$, where $\mathcal{O}$ is the observed-modality set. The dashed matrix in Fig. 3 illustrates the mechanism: an unobserved view is excluded from the denominator rather than being imputed or zeroed. The output is a weighted multi-view graph in which edge weight equals fused similarity.

### Structural-Entropy Community Discovery

<p align="center">
  <img src="docs/figures/poster-06-encoding-tree.png" alt="Encoding tree agglomerations: greedy merge rule ΔH > 0, stop at ΔH ≤ 0, leaves = users, optimal cut = communities" style="max-width: 1100px; width: 100%; display: block; margin: 1.5rem auto;">
</p>

The weighted graph is partitioned by **greedy agglomerative minimization of structural entropy** (Li & Pan, 2016). Each step merges the pair of communities whose union maximally reduces $\mathcal{H}(\mathcal{T})$ — a topological-information-theoretic cost that penalizes both small disconnected pieces and over-large lumps. The encoding tree (middle panel) makes the multi-scale structure explicit: every leaf is a single user, every internal node is a tentative community, and the global minimum of $\mathcal{H}$ identifies the natural scale (right panel). Crucially, the partition is **label-free**; no supervised loss is ever minimized.

### Post-Hoc Purity Interpretation

<p align="center">
  <img src="docs/figures/poster-07-community-archetypes.png" alt="Discovered community archetypes: red bot-rings, blue pure-human macro-cluster, grey peripheral fragments" style="max-width: 900px; width: 100%; display: block; margin: 1.5rem auto;">
</p>

The 898 discovered communities are grouped into four structural archetypes by their **post-hoc** label majority. Pure-human macro-regions are large and sparse; compact bot clusters are small, dense, and tightly interlinked; mixed transitional zones sit between them; sparse periphery groups isolated fragments. The global purity $\frac{1}{|\mathcal{C}|} \sum_C \max_y P(y \mid C) = 0.8643$ serves as a **validity indicator**, not a training objective — it tells us the discovered structure aligns with human labeling, but the labels never guided the search. This separation between *what is optimized* (structural entropy) and *what is reported* (label purity) is the methodological core of the project.

### Key Numbers (10k sampled TwiBot-22 subgraph)

| Metric | K-Means (baseline) | Weighted LPA | **Ours (Struct. Entropy)** |
|---|---|---|---|
| Lowest structural entropy $\mathcal{H}$ | 15.9861 | 13.8124 | **12.3537** |
| Largest community size | 2,734 | 1,402 | **283** |
| Density (largest comm.) | low | medium | **×55 over K-Means** |
| Clustering coefficient | low | medium | **×5 over K-Means** |
| Global purity (post-hoc) | — | — | **0.8643** |
| Communities | 8 | 162 | **898** |
| Labels used in optimization | ✓ | ✓ | **✗** |

**Headline result.** Lower structural entropy, finer-grained communities, and a structural-validity indicator above 0.86 — all without ever consulting one ground-truth label during the search.

---

## 🚀 Quickstart

### A. Explore the interactive demo

The dashboard is deployed on GitHub Pages — no setup required:

> **https://Majunrui524.github.io/BigData-Graduation-Innovation/**

Or run it locally:

```bash
cd project-code-implementation/demo
npm install
npm run dev        # → http://localhost:5173
```

### B. Run the Python pipeline (from the reproduce section above)

```bash
cd project-code-implementation

# 1. Environment
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env                                 # fill in your OpenAI-compatible API key

# 2. LLM-assisted semantic extraction (post-type + triplet compression)
bash scripts/run_full_api_pipeline.sh

# 3. Full 10k analysis: embeddings → graph → communities → evaluation
bash scripts/run_10k_late_results.sh
```

> **⚠️ Privacy & secrets:** `.env` holds your real API key and is git-ignored. Never commit it — only `.env.example` is versioned.

### C. Reproduce from the CLI

```bash
python -m twibot22_sampler.cli --help
# build-user-graph  extract-triplets  classify-post-types
# detect-communities  evaluate-community-purity  analyze-community-structure ...
```

### D. Run the tests

```bash
cd project-code-implementation
python -m pytest tests/ -q
```

---

## 🗄️ Dataset

Experiments use [TwiBot-22](https://github.com/LuoUndergradXJTU/TwiBot22), the largest Twitter bot-detection benchmark (~1M users, 170M relations). To keep this repository lightweight, the raw dataset is **not** bundled here (the full unpacked sample alone is > 2 GB):

1. Run `bash scripts/download_twibot22.sh` — fetches the official archive from the authors' Google Drive mirror (CC BY-NC-ND 4.0, academic research only; respect the Twitter Developer Agreement "Content redistribution" section).
2. Sample your subgraph with `python -m twibot22_sampler.cli sample --preset main --data-root data/twibot22_raw --work-root data/work_main --output-root data/samples/final_v1 --seed 42`.
3. Re-run the pipeline scripts.

The demo already ships with the precomputed 10k analysis bundles under `demo/public/data/10k/`, so the interactive dashboard works out of the box.

---

## 🛠️ Tech Stack

**Backend:** Python 3.10+ · NumPy · scikit-learn · Gensim · ijson · zero-dependency LLM client (sliding-window rate limiting, retries)

**Frontend:** React 18 · Vite · TypeScript · Tailwind CSS · ECharts · Sigma + Graphology · Framer Motion · Zustand · TanStack Table

---

## 🙏 Acknowledgement

- [TwiBot-22](https://github.com/LuoUndergradXJTU/TwiBot22) benchmark dataset
- The structural information theory line of work (Li & Pan, 2016) that inspired the encoding-tree formulation

---

## ⭐ Show Your Support

If this work resonates with your research, gave you an idea, or saved you some exploration time on the structural-entropy / community-detection rabbit hole — **a star on GitHub** is the easiest way to say thanks and helps the project reach other researchers working on bot detection, social-graph analysis, or unsupervised graph methods.

[![Star History Chart](https://api.star-history.com/svg?repos=Majunrui524/BigData-Graduation-Innovation&type=Date)](https://star-history.com/#Majunrui524/BigData-Graduation-Innovation&Date)

### 🙌 Contributing

Bug reports, dataset additions, alternative graph-construction strategies, and docs fixes are very welcome. Please open an issue first to discuss substantial changes; small fixes can go straight to a PR. The 26-module `pytest` suite under `project-code-implementation/tests/` is the fastest way to verify your changes don't regress core behavior.

---

## 📜 License

[MIT](LICENSE) © 2025 Anonymous
