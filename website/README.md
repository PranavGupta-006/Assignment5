# Assignment Visual Dashboard

This Vite app runs the three assignment Python scripts through a local dev-server API and renders their streamed JSON output as visual cards.

## Commands

```bash
npm run dev
npm run build
npm run lint
```

## Local API

The API is implemented in `vite.config.js`.

- `GET /api/scripts`: lists available assignment scripts.
- `GET /api/run/q1`: streams `q1_game_search.py --steps`.
- `GET /api/run/q2`: streams `q2_travel_planner.py --steps`.
- `GET /api/run/q3`: streams `q3_bayesian_networks.py --steps`.

Each Python script emits newline-delimited JSON. The frontend parses each line and chooses a visualization from the payload keys.

## Visualizations

- Game board and best-move highlighting for search algorithms.
- Search-effort bars for Minimax, Alpha-Beta, and heuristic Alpha-Beta.
- MCTS move visit counts and win rates.
- Destination ranking and cost cards for the travel planner.
- Knowledge graph sample diagram and KG tool catalog.
- Bayesian Network DAG, probability grids, inference comparison table, and MAP explanation.
