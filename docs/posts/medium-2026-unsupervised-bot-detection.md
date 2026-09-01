# Finding bot networks without any training labels: a structural-entropy walkthrough

> A practical tour of an open-source unsupervised social-bot detection pipeline. I walk through the problem, the key design choices, and the headline results on the TwiBot-22 benchmark, and end with a live demo you can try in your browser.
>
> **Repo**: [github.com/Majunrui524/BigData-Graduation-Innovation](https://github.com/Majunrui524/BigData-Graduation-Innovation)
> **Live demo**: [Majunrui524.github.io/BigData-Graduation-Innovation/#/detective](https://Majunrui524.github.io/BigData-Graduation-Innovation/#/detective)

---

## Why bother with unsupervised bot detection?

Almost every social-bot detector you'll find in production is a **supervised classifier**: somebody labelled a few thousand accounts as "human" or "bot", you train a binary model, you ship it. The problems with that approach are well known:

- Labelling is expensive and the labels go stale fast
- Bot operators change tactics weekly, models break weekly
- A binary "yes/no" tells you nothing about *how* the bots are coordinating

I wanted a system that needs **zero training labels** and surfaces not just "this account is suspicious" but "these 87 accounts form a coordinated ring".

## A counter-intuitive first result

On the TwiBot-22 benchmark (18,743 users, 202k tweets), I ran a head-to-head:

| Method | Communities | Largest community | Structural entropy | Post-hoc purity |
|---|---|---|---|---|
| K-Means | 898 | 2,734 | 15.99 | 0.863 |
| Weighted LPA | 241 | 5,798 | 14.20 | 0.864 |
| **Structural-entropy tree (ours)** | **898** | **283** | **12.35** | **0.864** |

The supervised method is not on this row — it doesn't use structure at all.

The interesting bit: **the structure-only method, with no labels anywhere in the optimisation, ends up with 88.6% of each community's members belonging to the same true class.** Structure is quietly leaking the truth.

## How it works: four views fused into one graph

A single signal is never enough. I build four independent views, then fuse them into a single weighted user graph.

**Content view**
Each post is compressed by an LLM into a subject-verb-object triplet, embedded into a vector. I also track the share of original / retweet / comment / link-share posts.
> Real users post mostly originals. Bots skew heavily toward retweets and link-shares.

**Behaviour view**
Posting volume, JavaScript string diversity, URL share, retweet ratio. Bots cluster at the extreme: very high volume, near-zero JS diversity.

**Temporal view**
Dynamic Time Warping distance between the account's 24h activity curve and a canonical human diurnal pattern.
> Humans have obvious day/night cycles. Bots are almost flat.

**Network view**
Jaccard similarity over followee sets. This exposes dense mutual-following clusters, the classic signature of a bot ring.

Each view alone gives a weak, noisy signal. The real signal is in their weighted combination.

## Adaptive late-fusion: don't punish accounts for missing data

Naive fusion gives each view a fixed weight. The problem: many accounts only have tweets, no follow graph. Others lack temporal data. A fixed-weight scheme treats missing views as zero signal and unfairly penalises those accounts.

The fix is per-account renormalisation:

$$
W_i = \frac{\hat{W}_i}{\sum_{j \in \mathcal{O}} \hat{W}_j}, \quad \mathcal{O} = \{j : j \text{ observed}\}
$$

We renormalise over the views the account actually has. Missing modalities are reabsorbed, not punished.

## Structural-entropy community discovery

With the weighted user graph in hand, the next question is "which accounts are structurally closer to each other". I use **structural-entropy minimisation**: a greedy agglomerative procedure that, at every step, merges the two subgraphs whose merge gives the largest drop in the graph's structural entropy.

The result is a **tree** — an encoding tree whose leaves are the discovered communities. On the 18,743-user sample, the tree produces 898 communities with post-hoc label purity 0.8643. Again: **no labels are used in the optimisation**.

## Four archetypes, not one binary

Unlike a classifier, the tree naturally surfaces four community archetypes:

| Archetype | Count | What it looks like |
|---|---|---|
| Pure-human macro | 118 | Large clusters, purity > 0.96 |
| Compact bot | 103 | Small, dense, bot-dominated rings |
| Mixed transitional | 215 | Human-bot mix, soft boundaries |
| Sparse peripheral | 462 | Weakly connected fragments |

You walk away with a *map* of the bot network, not just a verdict per account.

## Try it yourself

**1. Live demo (zero setup)**
Open [Majunrui524.github.io/BigData-Graduation-Innovation/#/detective](https://Majunrui524.github.io/BigData-Graduation-Innovation/#/detective):
- "Surprise me" → 11 random real accounts
- Click any account → full four-view evidence chain + final verdict
- Search by username / name / bio

All data is real, from TwiBot-22. Nothing is mocked.

**2. Verify the headline numbers in 30 seconds**
```bash
git clone https://github.com/Majunrui524/BigData-Graduation-Innovation
cd BigData-Graduation-Innovation
python project-code-implementation/tools/offline_reproduce.py
```
This recomputes every headline number (user count, community count, global purity, etc.) from the shipped JSON bundle and cross-checks it against `overview.json`. Exit code 0 = all 11 checks pass.

**3. Run the full pipeline**
You need an OpenAI-compatible embedding API. See the Quickstart in the [README](https://github.com/Majunrui524/BigData-Graduation-Innovation).

## Design trade-offs

- **Why not deep learning?** Supervised deep nets degrade fast as bot tactics shift. Structure-based methods are more robust to drift.
- **Why not just ask an LLM to classify?** Cost, latency, lack of interpretability. On 18k accounts, an LLM classifier takes hours and tens of dollars; the structural method takes seconds and cents.
- **Why TwiBot-22?** It's the largest publicly-available Twitter bot benchmark with carefully cleaned labels — perfect for *post-hoc* validation, even though I never use the labels during training.

## What's still hard

- Cross-platform transfer: the canonical "human" temporal curve is English-Twitter specific
- Real-time: this is offline batch analysis; ms-latency would need precomputed communities
- Privacy: all computation is local; no user data leaves the machine

## Closing

This started as my undergraduate thesis and I open-sourced the whole thing — code, docs, live demo. Issues, PRs and starred forks are all welcome.

If this was useful, the single best thing you can do is hit **⭐ Star** on the repo. That's the signal that gets the next person to find it.

**GitHub**: [Majunrui524/BigData-Graduation-Innovation](https://github.com/Majunrui524/BigData-Graduation-Innovation)
**Live demo**: [Majunrui524.github.io/BigData-Graduation-Innovation/#/detective](https://Majunrui524.github.io/BigData-Graduation-Innovation/#/detective)

---

> Tags: unsupervised-learning · community-detection · structural-entropy · social-bot · graph-representation-learning · anomaly-detection
