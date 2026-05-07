# Assignment 5: AI Search, Travel Planner KG, and Bayesian Networks

Submitted by: **Pranav Gupta**  
Roll No: **SE24UCSE020**

This project contains three AI assignment implementations plus a React frontend that runs them and visualizes the output.

## What Is Included

- `q1_game_search.py`: Tic-Tac-Toe AI with Minimax, Alpha-Beta pruning, depth-limited heuristic Alpha-Beta, and Monte-Carlo Tree Search.
- `q2_travel_planner.py`: AI travel planner using a small RDF-style triple store for destinations, attractions, cuisines, hotels, budget, seasons, and rules.
- `q3_bayesian_networks.py`: Bayesian Network example using the Asia chest-clinic network with `pgmpy`, exact inference, and MAP explanation.
- `website/`: React + Vite frontend that streams each Python script through a local API and shows modern visual cards.
- `.gitignore`: protects local junk, caches, build output, virtual environments, secrets, and OS/editor files.

## How It Works

The Python files can run in two modes:

- Normal testcase mode: prints tests and assertions in the terminal.
- Frontend mode: emits newline-delimited JSON using `--steps`.

The Vite frontend defines a small local API in `website/vite.config.js`.

- `GET /api/scripts`: returns the available assignment scripts.
- `GET /api/run/q1`: runs `python3 q1_game_search.py --steps`.
- `GET /api/run/q2`: runs `python3 q2_travel_planner.py --steps`.
- `GET /api/run/q3`: runs `python3 q3_bayesian_networks.py --steps`.

The React app reads that streamed output and renders boards, graphs, rankings, probabilities, and summary cards.

## Requirements

Install these first:

- Python 3.10 or newer
- Node.js 20 or newer
- npm, included with Node.js

Python package needed for Q3:

- `pgmpy`
- `numpy`

Q1 and Q2 use only the Python standard library.

## Install On macOS

Using Homebrew:

```bash
brew install python node
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pgmpy numpy
```

Install frontend dependencies:

```bash
cd website
npm install
cd ..
```

## Install On Windows

Install:

- Python from `https://www.python.org/downloads/`
- Node.js LTS from `https://nodejs.org/`

Then open PowerShell in the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pgmpy numpy
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Install frontend dependencies:

```powershell
cd website
npm install
cd ..
```

## Install On Linux

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip nodejs npm
```

If your distro ships an old Node.js version, install Node.js LTS from NodeSource or `nvm`.

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pgmpy numpy
```

Install frontend dependencies:

```bash
cd website
npm install
cd ..
```

## Run The Python Testcases

From the project root:

```bash
python q1_game_search.py
python q2_travel_planner.py
python q3_bayesian_networks.py
```

On some systems, use `python3` instead of `python`.

Expected coverage:

- Q1 tests terminal states, optimal Minimax, blocking/winning moves, Alpha-Beta equivalence, pruning, heuristic depth limits, MCTS convergence, and algorithm comparison.
- Q2 tests destination scoring, hotel selection, dietary rules, UNESCO queries, vegan cuisine queries, cost queries, and top attraction queries.
- Q3 tests model validation, prior marginals, posterior inference, explaining away, d-separation behaviour, Variable Elimination versus Belief Propagation, and MAP inference.

## Run The Frontend

From the project root:

```bash
cd website
npm run dev
```

Open the URL printed by Vite, usually:

```text
http://127.0.0.1:5173/
```

If that port is busy, Vite will automatically choose the next available port.

## Build The Frontend

```bash
cd website
npm run build
```

The production build is generated in `website/dist/`. It is ignored by git because it can be regenerated.

## Lint The Frontend

```bash
cd website
npm run lint
```

## Frontend Visualizations

The dashboard shows:

- Tic-Tac-Toe board and selected move.
- Search effort bars for nodes, pruning, saved nodes, and heuristic cutoffs.
- MCTS move confidence using visit counts and win rates.
- Travel destination ranking and estimated tour plan.
- Knowledge graph snapshot connecting destinations, attractions, cuisines, and hotels.
- Knowledge graph tool categories.
- Bayesian Network DAG, probability grids, inference comparison, and MAP result.

## Knowledge Graph Tools Covered

Graph databases:

- Neo4j
- Memgraph
- Amazon Neptune
- Azure Cosmos DB
- AllegroGraph
- GraphDB
- JanusGraph

Graph construction:

- LlamaIndex
- LangChain
- GliNER
- Infernotus
- ContextClue Graph Builder

Ontology/modeling:

- Protege
- TopBraid Composer

Visualization and personal KGs:

- Gephi
- Kumu
- Linkurious
- Obsidian Graph View
- TheBrain

## Bayesian Network Tools Covered

- `pgmpy`: Python modeling, CPDs, validation, exact inference.
- GeNIe / SMILE: visual Bayesian Network modeling.
- Netica: commercial Bayesian Network and decision network tooling.
- `bnlearn`: R package for learning and inference.
- Bayes Server: industrial Bayesian Network tooling and APIs.

## Git Safety Notes

The root `.gitignore` excludes:

- `.DS_Store`, `Thumbs.db`, and other OS junk.
- `node_modules/` and Vite build output.
- Python `__pycache__/`, `.venv/`, and test caches.
- `.env`, local secrets, keys, certificates, and credential JSON files.
- editor folders such as `.idea/` and most `.vscode/` files.

Before committing, you can check what will be included:

```bash
git status --short
```

Do not commit real `.env` files, API keys, private certificates, or local credential files.
