# Late Fusion Community Structure Explorer

Self-contained frontend demo for the 10k community-structure analysis.

It is part of the [bigdata-graduation-innovation](https://github.com/Majunrui524/bigdata-graduation-innovation) project. All required static assets and data bundles are already included under `public/`, so the folder can also be published standalone (e.g. to GitHub Pages) without any backend.

## What This Demo Is

This is a presentation-oriented web app for exploring:

- late-fusion graph construction
- structural-entropy minimization
- encoding-tree community discovery
- purity as a label-aware external reference

The main narrative is unsupervised community structure analysis rather than supervised classification.

## Pages

- `/`
  - Overview dashboard
  - Sample scale, graph size, entropy drop, structural quality indicators, and grouping-method summaries
- `/graph`
  - Community graph explorer
  - Node size reflects community size
  - Node color can switch between density and clustering coefficient
- `/communities`
  - Community table and inspector
  - Displays purity, density, clustering, average degree, encoding depth, archetype, and representative users
- `/compare`
  - Grouping-method comparison
  - Compares `K-Means`, `Weighted LPA`, and `Structural Entropy (Ours)` using structure-centered indicators
- `/errors`
  - Auxiliary diagnostic page retained for presentation support

## Tech Stack

- React 18
- Vite
- TypeScript
- React Router
- Tailwind CSS
- Framer Motion
- ECharts
- Graphology
- Sigma
- Zustand
- TanStack Table

## Project Structure

```text
demo/
├── public/
│   ├── data/10k/      # prebuilt JSON bundles
│   └── visuals/       # prebuilt visual assets
├── src/               # application source
├── index.html
├── package.json
└── README.md
```

## Run Locally

```bash
npm install
npm run dev
```

Then open:

- `http://localhost:5173/`

## Build

```bash
npm run build
```

Preview the production build:

```bash
npm run preview
```

## Notes

- This repository is static-data-driven. There is no live backend API.
- The data shown by the app is already bundled under `public/data/10k/`.
- The `Errors` page is secondary. The main presentation flow is `Overview -> Graph -> Communities -> Compare`.
- `demo_override.json` is kept as presentation metadata for selected display overrides. The app itself reads the already-generated JSON bundle in `public/data/10k/`.

## Recommended Screenshot Order

1. `Overview`
2. `Graph`
3. `Communities`
4. `Compare`
5. `Errors`
