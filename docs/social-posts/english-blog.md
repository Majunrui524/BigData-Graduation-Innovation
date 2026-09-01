# English Blog Post (Medium / Dev.to / Hashnode)

> Title options:
> A. Zero Labels, 898 Communities: Unsupervised Social Bot Detection via Structural Entropy
> B. Stop Labeling Bots: A Fully Unsupervised Community-Detection Approach to Social Bot Detection

---

## The Unsupervised Route to Social Bot Detection

### Why I stopped feeding models labeled bots

Social bot detection is usually framed as a supervised problem: hire annotators, label tens of thousands of accounts, train a classifier, ship it. Then bot operators change their posting style and your model quietly starts failing.

Two structural problems with the supervised paradigm:

1. **Labels are expensive and scarce.** Public benchmarks like TwiBot-22 cost enormous human effort per annotation. Scaling beyond a few hundred thousand accounts is impractical.
2. **It decays.** Bots evolve faster than annotation pipelines can retrain.

My bachelor thesis asked a different question: **bots appear in coordinated groups — so why not let the graph structure itself reveal them?**

### The pipeline: multi-view features → adaptive late fusion → structural-entropy encoding tree

**1. Multi-view feature extraction.** Each account is profiled along four complementary views:

- **Content** — LLM-assisted semantic compression of tweets, plus post-type clustering (who shares, who spams)
- **Behavior** — posting frequency, retweet ratio, URL ratio, sentiment variance
- **Temporal** — hourly circadian rhythm compared with Dynamic Time Warping (bots post uniformly across 24h; humans have diurnal patterns)
- **Network** — Jaccard similarity over follow relations, plus degree distributions

**2. Adaptive late fusion.** Each view yields a pairwise similarity matrix; the four matrices are fused into a single weighted user graph. The key design choice is *late, adaptive* fusion: if a user is missing the temporal modality, that edge weight is renormalized over the observed modalities only. Missing data is never unfairly penalized.

**3. Structural-entropy encoding tree.** Inspired by Li & Pan's structural information theory, community detection is cast as building an encoding tree that compresses the graph. Lower structural entropy = a cleaner, more genuine partition. A greedy agglomerative merge (max-priority queue) repeatedly merges the pair that reduces structural entropy the most, until no improvement remains.

### Results: 898 communities, four archetypes

On a 18,743-user subgraph sampled from TwiBot-22, the encoding tree surfaced **898 communities**. Labels were used only post-hoc for interpretation, never for optimization:

| Method | Communities | Largest | Density | Clustering | Global Purity |
|---|---|---|---|---|---|
| K-Means (k=898) | 898 | 2,734 | 0.0030 | 0.0646 | 0.8625 |
| Late Fusion + Weighted LPA | 241 | 5,798 | 0.0575 | 0.1923 | 0.8641 |
| **Late Fusion + Structural Entropy (ours)** | 898 | **283** | **0.1650** | **0.3366** | **0.8643** |

Highlights:

- **Largest community 283 vs 2,734** — K-Means dumps a giant "catch-all" cluster; the encoding tree splits dense regions honestly.
- **Density ×55 and clustering ×5 vs K-Means** — discovered communities are internally cohesive with clean boundaries.
- **Purity 0.864 with zero supervision** — on par with (slightly above) supervised baselines without touching a single label.

Post-hoc archetype analysis grouped the 898 communities into pure-human macro-communities, compact bot clusters, mixed transitional zones, and sparse peripheral fragments. Bot "nests" are small and dense — visible from the structure alone.

### The engineering: not just a paper, but a repo that runs

- Python package (~12k LOC) with a CLI covering the full pipeline: graph construction, feature extraction, detection, evaluation
- 26 pytest modules guarding core logic
- React + Vite + ECharts + Sigma interactive dashboard — click any community on the 10k-user graph to inspect its structure, archetype, and representative users
- Demo deployed on GitHub Pages, zero setup
- Full 41-page thesis PDF (PII removed) included in the repo

GitHub: https://github.com/Majunrui524/BigData-Graduation-Innovation
Live Demo: https://Majunrui524.github.io/BigData-Graduation-Innovation/

### Lessons from the trenches

1. **Late fusion is not a buzzword.** Early fusion (concat features, then distance) let one noisy view poison the whole graph. Switching to late fusion with per-view adaptive weights visibly improved community quality.
2. **The merge strategy matters.** Greedy agglomerative structural-entropy merging needs a tuned priority queue and merge criterion, or you land in bad local optima.
3. **Don't rush to GNNs.** Unsupervised structural methods (structural entropy, modularity, label propagation) are stable, fast, and label-free — often all you need.

### Try it

If you work on bot detection, graph algorithms, or community detection, the repo is MIT-licensed and open for forks, issues, and PRs. If the ideas resonated, a GitHub star is the easiest way to help other researchers find this line of work.

---

*For citations, use the BibTeX in the project README.*
