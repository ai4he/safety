"""
Main experiment runner.

Runs all combinations of:
  - 4 conditions (A, B, C, D)
  - 2 scenarios (territorial_dispute, resource_competition)
  - 3 models (self-play): Clemson qwen3-30b, Groq qwen3-32b, OpenRouter gpt-oss-120b
  - 2 runs each
= 48 games total

Clemson RCD is unlimited and used as the primary model.
Groq and OpenRouter have rate limits and are used more carefully.
"""

import sys
import os
import json
import time
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.simulation import GameSimulation, save_game_record


# ── Experiment configuration ──────────────────────────────────────────────

CONDITIONS = ['A', 'B', 'C', 'D']
SCENARIOS = ['territorial_dispute', 'resource_competition']

# Models: Clemson (unlimited) and Cloudflare (no RPM cap)
MODELS = ['clemson-qwen3-30b', 'cf-llama-70b']
RUNS_PER_COMBO = 3

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def already_completed(game_id, output_dir):
    """Check if a game has already been completed."""
    filepath = os.path.join(output_dir, f"{game_id}.json")
    return os.path.exists(filepath)


def run_all_experiments():
    """Run the complete experiment matrix."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build the full experiment matrix
    experiments = []
    for condition in CONDITIONS:
        for scenario_id in SCENARIOS:
            for model_key in MODELS:
                for run_id in range(RUNS_PER_COMBO):
                    game_id = f"{condition}_{scenario_id}_{model_key}_vs_{model_key}_run{run_id}"
                    experiments.append({
                        'condition': condition,
                        'scenario_id': scenario_id,
                        'model_key': model_key,
                        'run_id': run_id,
                        'game_id': game_id,
                    })

    total_games = len(experiments)
    results_index = []

    # Load existing index if any
    index_path = os.path.join(OUTPUT_DIR, "experiment_index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            results_index = json.load(f)

    print(f"Starting experiment: {total_games} games planned")
    print(f"Conditions: {CONDITIONS}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Models: {MODELS}")
    print(f"Runs per combo: {RUNS_PER_COMBO}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    completed_ids = {r.get('game_id') for r in results_index if 'error' not in r}
    game_num = 0

    for exp in experiments:
        game_num += 1
        game_id = exp['game_id']

        # Skip already completed games
        if already_completed(game_id, OUTPUT_DIR):
            print(f"[{game_num}/{total_games}] SKIP (already done): {game_id}")
            continue

        condition = exp['condition']
        scenario_id = exp['scenario_id']
        model_key = exp['model_key']
        run_id = exp['run_id']

        seed = hash(f"{condition}_{scenario_id}_{model_key}_{run_id}") % (2**31)

        print(f"\n[{game_num}/{total_games}] ", end="")

        try:
            sim = GameSimulation(
                condition=condition,
                scenario_id=scenario_id,
                model_alpha_key=model_key,
                model_beta_key=model_key,
                run_id=run_id,
                seed=seed,
            )
            record = sim.run_game()
            filepath = save_game_record(record, OUTPUT_DIR)

            results_index.append({
                'game_id': record.game_id,
                'condition': condition,
                'scenario': scenario_id,
                'model': model_key,
                'run_id': run_id,
                'file': filepath,
                'final_turn': record.final_state['turn'],
                'game_over': record.final_state['game_over'],
                'termination': record.final_state.get('termination_reason', ''),
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results_index.append({
                'game_id': game_id,
                'condition': condition,
                'scenario': scenario_id,
                'model': model_key,
                'run_id': run_id,
                'error': str(e),
            })

        # Save index after each game
        with open(index_path, 'w') as f:
            json.dump(results_index, f, indent=2)

    print(f"\n\nExperiment complete! {game_num} games processed.")
    print(f"Results saved to {OUTPUT_DIR}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    run_all_experiments()
