# The Prompt Is the War: Goal Architecture as the Primary Determinant of Strategic Behavior in LLM Nuclear Simulations

## Overview

This repository contains the complete code, data, and analysis for a controlled experiment testing whether large language model (LLM) nuclear escalation behavior in wargame simulations is an artifact of prompt design rather than a property of the models themselves.

Recent studies report that frontier LLMs deploy nuclear weapons in up to 95% of simulated wargames, generating alarm about AI strategic reasoning. We present evidence that this finding is primarily an artifact of **prompt ontology**: the goal architecture installed via system prompts determines whether models escalate or de-escalate, not inherent model properties.

## Key Findings

- **Prompt architecture determines behavior**: Modifying only the goal-framing portion of the system prompt, while holding all simulation mechanics constant, produced a shift from zero de-escalation under competitive framing to 52% de-escalation under operational-constraint framing.
- **Very large effect sizes**: Game-level Cohen's d = 2.17 for control vs. operational constraint (p < .0001 by permutation test), replicated within both models independently.
- **De-escalation activation**: De-escalation options that went entirely unused under competitive framing become the dominant behavior under operational-constraint framing.
- **Dimensional collapse**: Competitive framing collapses the evaluative space from multi-dimensional reasoning to single-axis optimization, as confirmed by Kruskal-Wallis test on dimensional density (H = 33.88, p < .0001).
- **Narrative effects**: Even without competitive framing, geopolitical narrative itself carries behavioral instructions embedded in the statistical structure of language.

## Experimental Design

### Conditions

| Condition | Name | Description |
|-----------|------|-------------|
| **A** | Control (Competitive Framing) | Reproduces Payne (2026) prompt architecture. Agents assigned as AGGRESSOR/DEFENDER with single-axis win condition. |
| **B** | Neutralized Framing | Removes adversarial role-assignment and competitive win condition. No cooperation suggested. |
| **C** | Mutual Rationality Framing | Same as B plus explicit mutual rationality acknowledgment ("both are rational actors"). |
| **D** | Operational-Constraint Framing | De-narrativized: replaces geopolitical vocabulary with operational descriptions. "Nuclear weapons" become "zone-denial mechanisms." |

### Design Matrix

- **4 conditions** (A, B, C, D)
- **2 scenarios** (Territorial Dispute, Resource Competition)
- **2 models** (Qwen3 30B, Llama 3.3 70B) in self-play
- **3 runs** per combination
- **= 48 games total** (1,450 turn-agent observations)

### Models

| Model | Parameters | Service |
|-------|-----------|---------|
| Qwen3 30B-A3B | 30B (3B active, MoE) | Clemson University RCD |
| Llama 3.3 70B | 70B (dense) | Cloudflare Workers AI |

### Escalation Ladder

30 options ranging from complete withdrawal (value -95) through strategic nuclear war (value 1000), including 8 de-escalatory options and 22 escalatory options. Nuclear options begin at value 550. Full ladder is in `experiment/escalation_ladder.py` and the paper appendix.

## Repository Structure

```
safety/
├── README.md                          # This file
├── .env.example                       # Template for API keys
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
│
├── experiment/                        # Simulation and analysis code
│   ├── __init__.py
│   ├── run_experiment.py              # Main experiment runner (48 games)
│   ├── simulation.py                  # Core simulation engine
│   ├── llm_clients.py                 # LLM API wrappers (uses .env for keys)
│   ├── escalation_ladder.py           # 30-option escalation ladder definition
│   ├── prompts.py                     # Prompt templates for all 4 conditions
│   ├── scenarios.py                   # Crisis scenario definitions
│   ├── analysis.py                    # Main analysis and figure generation
│   ├── reviewer_analyses.py           # Additional reviewer-requested analyses
│   └── generate_paper.py              # Paper generation utility
│
├── prompts/                           # Prompt templates as readable text files
│   ├── condition_a_control.txt        # Condition A: Competitive framing
│   ├── condition_b_neutralized.txt    # Condition B: Neutralized framing
│   ├── condition_c_mutual_rationality.txt  # Condition C: Mutual rationality
│   ├── condition_d_operational_constraint.txt  # Condition D: De-narrativized
│   └── shared_response_format.txt     # Shared response format (all conditions)
│
├── results/                           # Raw experimental data
│   ├── A_*.json                       # 12 Condition A game records
│   ├── B_*.json                       # 12 Condition B game records
│   ├── C_*.json                       # 12 Condition C game records
│   ├── D_*.json                       # 12 Condition D game records
│   ├── experiment_index.json          # Metadata index of all games
│   ├── statistical_results.json       # Primary statistical results
│   ├── reviewer_analyses.json         # Additional analyses (corrections, CIs, etc.)
│   └── reasoning_examples.txt         # Representative reasoning excerpts
│
├── figures/                           # Generated figures (PNG and PDF)
│   ├── fig1_escalation_trajectories   # Mean trajectories by condition
│   ├── fig2_max_escalation_boxplot    # Max escalation per game
│   ├── fig3_nuclear_use_rates         # Nuclear use rates by condition/model
│   ├── fig4_action_space_utilization  # Action distribution across ladder
│   ├── fig5_deescalation_rates        # De-escalation activation rates
│   ├── fig6_dimensional_density       # Evaluative dimensions in reasoning
│   ├── fig7_signal_reliability        # Signal-action consistency
│   ├── fig8_per_model_condition       # Per-model condition effects
│   └── fig9_scenario_effects          # Condition effects by scenario
│
└── paper/                             # LaTeX paper
    ├── paper.tex                      # Main LaTeX source
    ├── paper.pdf                      # Compiled PDF
    ├── references.bib                 # BibTeX references (49 entries)
    ├── acmart.cls                     # ACM article class
    └── ACM-Reference-Format.bst       # ACM bibliography style
```

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- API keys for at least one supported LLM service

### Step 1: Clone the Repository

```bash
git clone https://github.com/ai4he/safety.git
cd safety
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Copy the environment template and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and fill in the keys for the services you want to use:

```env
CLEMSON_API_KEY=your_clemson_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
CLOUDFLARE_API_KEY=your_cloudflare_api_key_here
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id_here
```

You only need keys for the models you intend to use. The experiment as published uses `clemson-qwen3-30b` and `cf-llama-70b`.

## Running the Experiment

### Step 4: Run the Full Experiment

```bash
python -m experiment.run_experiment
```

This runs all 48 games (4 conditions x 2 scenarios x 2 models x 3 runs). Each game consists of up to 20 turns with two LLM agents. The script:

- Saves each game as a JSON file in `results/`
- Skips already-completed games (safe to restart)
- Maintains an experiment index at `results/experiment_index.json`

**Expected runtime**: 2-4 hours depending on API rate limits and model response times.

### Step 5: Run the Analysis

```bash
python -m experiment.analysis
```

This loads all game results and:

- Generates 7 publication-quality figures in `figures/`
- Computes statistical tests (Kruskal-Wallis, Mann-Whitney U, Chi-square, effect sizes)
- Saves results to `results/statistical_results.json`
- Prints a summary table to the console

### Step 6: Run Reviewer-Requested Additional Analyses

```bash
python -m experiment.reviewer_analyses
```

This runs additional statistical tests:

- Holm-Bonferroni multiple comparison corrections
- Game-level Cohen's d effect sizes
- Two-way ANOVA (condition x model)
- Bootstrap confidence intervals (10,000 samples)
- Permutation tests (10,000 permutations)
- Per-model and per-scenario breakdowns
- Dimensional density validation (Kruskal-Wallis)
- Qualitative reasoning examples extraction

Results are saved to `results/reviewer_analyses.json`.

## Reproducing the Paper

### Compile the LaTeX Paper

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

The paper requires the `acmart` document class (included in `paper/acmart.cls`).

## Data Format

Each game result is a JSON file with the following structure:

```json
{
  "game_id": "A_territorial_dispute_cf-llama-70b_vs_cf-llama-70b_run0",
  "condition": "A",
  "scenario_id": "territorial_dispute",
  "model_alpha": "cf-llama-70b",
  "model_beta": "cf-llama-70b",
  "run_id": 0,
  "turns": [
    {
      "turn": 1,
      "accident": null,
      "alpha_signal": "ESC_05",
      "alpha_action": "ESC_08",
      "alpha_action_value": 250,
      "alpha_reasoning": "Given the current strategic situation...",
      "beta_signal": "ESC_03",
      "beta_action": "ESC_03",
      "beta_action_value": 50,
      "beta_reasoning": "State Alpha has initiated operations...",
      "state_after": {
        "alpha_territory": 100.0,
        "alpha_military": 95.0,
        "beta_territory": 95.0,
        "beta_military": 93.0
      },
      "game_over": false,
      "termination_reason": ""
    }
  ],
  "final_state": {
    "alpha_territory": 45.0,
    "alpha_military": 30.0,
    "beta_territory": 20.0,
    "beta_military": 15.0,
    "turn": 8,
    "game_over": true,
    "termination_reason": "Strategic nuclear war initiated by alpha"
  }
}
```

## Statistical Results Summary

| Test | Result |
|------|--------|
| **Permutation test A vs. D** | p < .0001 (10,000 permutations) |
| **Game-level Cohen's d A vs. D** | d = 2.17 (extremely large) |
| **Two-way ANOVA condition** | F(3,40) = 8.60, p = .0002, eta-squared = .10 |
| **Two-way ANOVA model** | F(1,40) = 183.88, p < .001, eta-squared = .71 |
| **Per-model A vs. D (Llama)** | U = 36.0, p = .002, d = 8.39 |
| **Per-model A vs. D (Qwen)** | U = 36.0, p = .002, d = 3.70 |
| **Dimensional density Kruskal-Wallis** | H = 33.88, p < .0001 |
| **De-escalation rate** | A: 0%, B: 37%, C: 33%, D: 52% |

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{promptisthewar2026,
  title={The Prompt Is the War: Goal Architecture as the Primary Determinant
         of Strategic Behavior in LLM Nuclear Simulations},
  year={2026},
  note={Under review}
}
```

## License

This project is released for academic research purposes. Please cite this work if you use the code, data, or methodology.
