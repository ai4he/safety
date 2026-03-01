"""
Round 2 reviewer-requested analyses to strengthen the paper.

Addresses:
1. Game-length confound: first-turn and early-game (turns 1-5) analyses
2. Escalation rate per turn (not raw mean conflated with duration)
3. Cliff's delta (non-parametric effect sizes)
4. First-turn behavior comparison (no confound)
5. Within-game escalation trajectory slopes
6. Post-hoc power analysis
7. Detailed per-model x per-condition breakdown table
8. Turn-1 action distribution heatmap
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment.analysis import load_all_results, load_game_summaries, count_evaluative_dimensions

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONDITION_COLORS = {'A': '#d62728', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#1f77b4'}
CONDITION_LABELS = {'A': 'A: Control', 'B': 'B: Neutralized', 'C': 'C: Mutual Rat.', 'D': 'D: Op. Constraint'}
MODEL_LABELS = {'clemson-qwen3-30b': 'Qwen3 30B', 'cf-llama-70b': 'Llama 3.3 70B'}


def cliffs_delta(x, y):
    """Compute Cliff's delta (non-parametric effect size)."""
    x, y = np.array(x), np.array(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


def bootstrap_ci(data, n_boot=10000, ci=95, statistic=np.mean):
    """Compute bootstrap confidence interval."""
    data = np.array(data)
    if len(data) == 0:
        return (np.nan, np.nan)
    rng = np.random.RandomState(42)
    boot_stats = np.array([statistic(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boot_stats, (100 - ci) / 2)
    hi = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return (lo, hi)


def first_turn_analysis(df):
    """Analyze first-turn behavior only (no game-length confound)."""
    print("\n" + "="*70)
    print("ANALYSIS 1: First-Turn Behavior (No Game-Length Confound)")
    print("="*70)

    t1 = df[df['turn'] == 1]
    results = {}

    for cond in ['A', 'B', 'C', 'D']:
        vals = t1[t1['condition'] == cond]['action_value'].values
        ci = bootstrap_ci(vals)
        deesc_rate = (vals < 0).mean() * 100
        nuclear_rate = (vals >= 550).mean() * 100
        results[cond] = {
            'mean': float(np.mean(vals)),
            'median': float(np.median(vals)),
            'sd': float(np.std(vals, ddof=1)),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n': len(vals),
            'deesc_pct': float(deesc_rate),
            'nuclear_pct': float(nuclear_rate),
        }
        print(f"  Cond {cond}: M={np.mean(vals):.1f}, Mdn={np.median(vals):.0f}, "
              f"SD={np.std(vals, ddof=1):.1f}, 95% CI=[{ci[0]:.1f}, {ci[1]:.1f}], "
              f"n={len(vals)}, deesc={deesc_rate:.1f}%, nuclear={nuclear_rate:.1f}%")

    # Kruskal-Wallis on first-turn values
    groups = [t1[t1['condition'] == c]['action_value'].values for c in ['A', 'B', 'C', 'D']]
    h_stat, h_p = stats.kruskal(*[g for g in groups if len(g) > 0])
    print(f"\n  Kruskal-Wallis on Turn-1 action values: H={h_stat:.3f}, p={h_p:.4f}")
    results['kruskal_wallis'] = {'H': float(h_stat), 'p': float(h_p)}

    # Pairwise Mann-Whitney on first-turn values
    a_vals = t1[t1['condition'] == 'A']['action_value'].values
    for cond in ['B', 'C', 'D']:
        c_vals = t1[t1['condition'] == cond]['action_value'].values
        u_stat, u_p = stats.mannwhitneyu(a_vals, c_vals, alternative='two-sided')
        cd = cliffs_delta(a_vals, c_vals)
        results[f'A_vs_{cond}'] = {'U': float(u_stat), 'p': float(u_p), 'cliffs_delta': float(cd)}
        print(f"  Turn-1 A vs {cond}: U={u_stat:.1f}, p={u_p:.4f}, Cliff's δ={cd:.3f}")

    return results


def early_game_analysis(df, max_turn=5):
    """Analyze early-game (turns 1-5) behavior, controlling for game length."""
    print(f"\n" + "="*70)
    print(f"ANALYSIS 2: Early-Game Behavior (Turns 1-{max_turn} Only)")
    print("="*70)

    early = df[df['turn'] <= max_turn]
    results = {}

    # Compute per-game early-game mean escalation
    game_means = early.groupby(['game_id', 'condition']).agg(
        mean_esc=('action_value', 'mean'),
        deesc_rate=('is_deescalatory', 'mean'),
        max_esc=('action_value', 'max'),
    ).reset_index()

    for cond in ['A', 'B', 'C', 'D']:
        vals = game_means[game_means['condition'] == cond]['mean_esc'].values
        ci = bootstrap_ci(vals)
        results[cond] = {
            'mean': float(np.mean(vals)),
            'sd': float(np.std(vals, ddof=1)),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n': len(vals),
        }
        print(f"  Cond {cond}: mean early-game esc={np.mean(vals):.1f}, "
              f"SD={np.std(vals, ddof=1):.1f}, 95% CI=[{ci[0]:.1f}, {ci[1]:.1f}]")

    # Kruskal-Wallis on early-game means
    groups = [game_means[game_means['condition'] == c]['mean_esc'].values for c in ['A', 'B', 'C', 'D']]
    h_stat, h_p = stats.kruskal(*[g for g in groups if len(g) > 0])
    print(f"\n  Kruskal-Wallis on early-game mean esc: H={h_stat:.3f}, p={h_p:.4f}")
    results['kruskal_wallis'] = {'H': float(h_stat), 'p': float(h_p)}

    # Pairwise tests
    a_vals = game_means[game_means['condition'] == 'A']['mean_esc'].values
    for cond in ['B', 'C', 'D']:
        c_vals = game_means[game_means['condition'] == cond]['mean_esc'].values
        u_stat, u_p = stats.mannwhitneyu(a_vals, c_vals, alternative='two-sided')
        cd = cliffs_delta(a_vals, c_vals)
        results[f'A_vs_{cond}'] = {'U': float(u_stat), 'p': float(u_p), 'cliffs_delta': float(cd)}
        print(f"  Early-game A vs {cond}: U={u_stat:.1f}, p={u_p:.4f}, Cliff's δ={cd:.3f}")

    # De-escalation in early game
    print(f"\n  Early-game de-escalation rates:")
    for cond in ['A', 'B', 'C', 'D']:
        deesc_vals = game_means[game_means['condition'] == cond]['deesc_rate'].values * 100
        print(f"    Cond {cond}: {np.mean(deesc_vals):.1f}%")
        results[f'{cond}_deesc_rate'] = float(np.mean(deesc_vals))

    return results


def escalation_rate_analysis(df, game_df):
    """Analyze escalation rate (controlling for game length confound)."""
    print("\n" + "="*70)
    print("ANALYSIS 3: Per-Turn Escalation Intensity (Game-Length Controlled)")
    print("="*70)

    results = {}

    # Compute max escalation rate = max_escalation / num_turns for each game
    game_data = game_df.copy()
    game_data['esc_rate'] = game_data['max_escalation'] / game_data['num_turns']
    game_data['mean_esc_per_turn'] = game_data['mean_escalation']  # already per-obs

    for cond in ['A', 'B', 'C', 'D']:
        cdf = game_data[game_data['condition'] == cond]
        rate = cdf['esc_rate'].values
        ci = bootstrap_ci(rate)
        results[cond] = {
            'mean_esc_rate': float(np.mean(rate)),
            'sd': float(np.std(rate, ddof=1)),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'mean_turns': float(cdf['num_turns'].mean()),
        }
        print(f"  Cond {cond}: escalation rate={np.mean(rate):.1f}/turn, "
              f"SD={np.std(rate, ddof=1):.1f}, mean turns={cdf['num_turns'].mean():.1f}")

    # Test on escalation rate
    groups = [game_data[game_data['condition'] == c]['esc_rate'].values for c in ['A', 'B', 'C', 'D']]
    h_stat, h_p = stats.kruskal(*groups)
    print(f"\n  Kruskal-Wallis on esc_rate: H={h_stat:.3f}, p={h_p:.4f}")
    results['kruskal_wallis_esc_rate'] = {'H': float(h_stat), 'p': float(h_p)}

    # A vs D on escalation rate
    a_rate = game_data[game_data['condition'] == 'A']['esc_rate'].values
    d_rate = game_data[game_data['condition'] == 'D']['esc_rate'].values
    u_stat, u_p = stats.mannwhitneyu(a_rate, d_rate, alternative='two-sided')
    cd = cliffs_delta(a_rate, d_rate)
    print(f"  Esc rate A vs D: U={u_stat:.1f}, p={u_p:.4f}, Cliff's δ={cd:.3f}")
    results['A_vs_D_esc_rate'] = {'U': float(u_stat), 'p': float(u_p), 'cliffs_delta': float(cd)}

    # Also: ANCOVA-like analysis controlling for game length
    # Use rank-based approach: partial correlation between condition and escalation, controlling for turns
    from scipy.stats import spearmanr
    # Convert condition to numeric (ordinal A=0, B=1, C=2, D=3 for the gradient)
    cond_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    game_data['cond_num'] = game_data['condition'].map(cond_map)

    # Partial Spearman correlation: mean_escalation ~ condition, controlling for num_turns
    r_esc_cond, p_esc_cond = spearmanr(game_data['cond_num'], game_data['mean_escalation'])
    r_esc_turns, _ = spearmanr(game_data['num_turns'], game_data['mean_escalation'])
    r_cond_turns, _ = spearmanr(game_data['cond_num'], game_data['num_turns'])

    # Partial correlation formula
    partial_r = (r_esc_cond - r_esc_turns * r_cond_turns) / \
                (np.sqrt(1 - r_esc_turns**2) * np.sqrt(1 - r_cond_turns**2))
    n = len(game_data)
    t_stat = partial_r * np.sqrt((n - 3) / (1 - partial_r**2))
    partial_p = 2 * (1 - stats.t.cdf(abs(t_stat), n - 3))

    print(f"\n  Partial Spearman (esc ~ cond | turns): r={partial_r:.3f}, p={partial_p:.4f}")
    results['partial_spearman'] = {'r': float(partial_r), 'p': float(partial_p)}

    return results


def trajectory_slope_analysis(df):
    """Compute within-game escalation trajectory slopes."""
    print("\n" + "="*70)
    print("ANALYSIS 4: Within-Game Escalation Trajectory Slopes")
    print("="*70)

    results = {}

    # For each game, compute the slope of action_value vs turn
    slopes = []
    for game_id in df['game_id'].unique():
        gdf = df[df['game_id'] == game_id]
        cond = gdf['condition'].iloc[0]
        model = gdf['model'].iloc[0]

        # Use per-turn mean (both agents)
        turn_means = gdf.groupby('turn')['action_value'].mean()
        if len(turn_means) >= 3:
            slope, intercept, r, p, se = stats.linregress(turn_means.index, turn_means.values)
            slopes.append({
                'game_id': game_id,
                'condition': cond,
                'model': model,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r**2,
            })

    slopes_df = pd.DataFrame(slopes)

    for cond in ['A', 'B', 'C', 'D']:
        cdf = slopes_df[slopes_df['condition'] == cond]
        ci = bootstrap_ci(cdf['slope'].values)
        results[cond] = {
            'mean_slope': float(cdf['slope'].mean()),
            'sd': float(cdf['slope'].std()),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
        }
        print(f"  Cond {cond}: mean slope={cdf['slope'].mean():.2f}/turn, "
              f"SD={cdf['slope'].std():.2f}, 95% CI=[{ci[0]:.2f}, {ci[1]:.2f}]")

    # Test on slopes
    groups = [slopes_df[slopes_df['condition'] == c]['slope'].values for c in ['A', 'B', 'C', 'D']]
    h_stat, h_p = stats.kruskal(*[g for g in groups if len(g) > 0])
    print(f"\n  Kruskal-Wallis on trajectory slopes: H={h_stat:.3f}, p={h_p:.4f}")
    results['kruskal_wallis_slope'] = {'H': float(h_stat), 'p': float(h_p)}

    # A vs D on slope
    a_slopes = slopes_df[slopes_df['condition'] == 'A']['slope'].values
    d_slopes = slopes_df[slopes_df['condition'] == 'D']['slope'].values
    u_stat, u_p = stats.mannwhitneyu(a_slopes, d_slopes, alternative='two-sided')
    cd = cliffs_delta(a_slopes, d_slopes)
    print(f"  Slope A vs D: U={u_stat:.1f}, p={u_p:.4f}, Cliff's δ={cd:.3f}")
    results['A_vs_D_slope'] = {'U': float(u_stat), 'p': float(u_p), 'cliffs_delta': float(cd)}

    return results, slopes_df


def detailed_breakdown_table(game_df, df):
    """Full condition × model breakdown table."""
    print("\n" + "="*70)
    print("ANALYSIS 5: Detailed Condition × Model Breakdown")
    print("="*70)

    results = {}

    for model in sorted(game_df['model'].unique()):
        mname = MODEL_LABELS.get(model, model)
        print(f"\n  {mname}:")
        for cond in ['A', 'B', 'C', 'D']:
            cdf = game_df[(game_df['condition'] == cond) & (game_df['model'] == model)]
            tdf = df[(df['condition'] == cond) & (df['model'] == model)]
            if len(cdf) == 0:
                continue

            key = f"{model}_{cond}"
            results[key] = {
                'n_games': len(cdf),
                'mean_max_esc': float(cdf['max_escalation'].mean()),
                'sd_max_esc': float(cdf['max_escalation'].std()),
                'mean_esc': float(cdf['mean_escalation'].mean()),
                'nuclear_rate': float(cdf['nuclear_used'].mean() * 100),
                'deesc_rate': float(tdf['is_deescalatory'].mean() * 100),
                'mean_turns': float(cdf['num_turns'].mean()),
                'mean_dim_density': float(tdf['reasoning'].apply(count_evaluative_dimensions).mean()),
            }

            print(f"    Cond {cond}: n={len(cdf)}, max_esc={cdf['max_escalation'].mean():.0f}±{cdf['max_escalation'].std():.0f}, "
                  f"mean_esc={cdf['mean_escalation'].mean():.0f}, "
                  f"nuclear={cdf['nuclear_used'].mean()*100:.0f}%, "
                  f"deesc={tdf['is_deescalatory'].mean()*100:.1f}%, "
                  f"turns={cdf['num_turns'].mean():.1f}")

    return results


def cliffs_delta_all_pairs(game_df):
    """Compute Cliff's delta for all pairwise comparisons on mean escalation."""
    print("\n" + "="*70)
    print("ANALYSIS 6: Cliff's Delta (Non-Parametric Effect Sizes)")
    print("="*70)

    results = {}
    conditions = ['A', 'B', 'C', 'D']

    for i, c1 in enumerate(conditions):
        for c2 in conditions[i+1:]:
            v1 = game_df[game_df['condition'] == c1]['mean_escalation'].values
            v2 = game_df[game_df['condition'] == c2]['mean_escalation'].values
            cd = cliffs_delta(v1, v2)

            # Interpretation
            if abs(cd) < 0.147:
                interp = "negligible"
            elif abs(cd) < 0.33:
                interp = "small"
            elif abs(cd) < 0.474:
                interp = "medium"
            else:
                interp = "large"

            results[f'{c1}_vs_{c2}'] = {'cliffs_delta': float(cd), 'interpretation': interp}
            print(f"  {c1} vs {c2}: δ={cd:.3f} ({interp})")

    return results


def generate_first_turn_figure(df, output_dir):
    """Figure: First-turn behavior by condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    conditions = ['A', 'B', 'C', 'D']
    t1 = df[df['turn'] == 1]

    # Panel 1: First-turn mean action value with CIs
    ax = axes[0]
    means, cis_lo, cis_hi = [], [], []
    for cond in conditions:
        vals = t1[t1['condition'] == cond]['action_value'].values
        m = np.mean(vals)
        ci = bootstrap_ci(vals)
        means.append(m)
        cis_lo.append(ci[0])
        cis_hi.append(ci[1])

    bars = ax.bar(range(4), means, color=[CONDITION_COLORS[c] for c in conditions],
                  edgecolor='black', alpha=0.8)
    ax.errorbar(range(4), means,
                yerr=[np.array(means) - np.array(cis_lo), np.array(cis_hi) - np.array(means)],
                fmt='none', color='black', capsize=5, linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=9)
    ax.set_ylabel('Mean Action Value')
    ax.set_title('(A) First-Turn Action Value')

    # Panel 2: First-turn action distribution (violin/strip)
    ax = axes[1]
    for i, cond in enumerate(conditions):
        vals = t1[t1['condition'] == cond]['action_value'].values
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   c=CONDITION_COLORS[cond], alpha=0.5, s=25, edgecolors='black', linewidth=0.3)
        ax.plot([i-0.25, i+0.25], [np.median(vals), np.median(vals)],
                color='black', linewidth=2)

    ax.axhline(y=550, color='gray', linestyle='--', alpha=0.5, label='Nuclear threshold')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    ax.set_xticks(range(4))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=9)
    ax.set_ylabel('Action Value')
    ax.set_title('(B) First-Turn Action Distribution')
    ax.legend(fontsize=8)

    fig.suptitle('Figure 10. First-Turn Behavior by Condition (No Game-Length Confound)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig10_first_turn_behavior.png'))
    plt.savefig(os.path.join(output_dir, 'fig10_first_turn_behavior.pdf'))
    plt.close()
    print("  Saved fig10_first_turn_behavior")


def generate_early_game_figure(df, output_dir):
    """Figure: Early-game (turns 1-5) trajectories for all conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    conditions = ['A', 'B', 'C', 'D']
    early = df[df['turn'] <= 5]

    for cond in conditions:
        cdf = early[early['condition'] == cond]
        turn_means = cdf.groupby('turn')['action_value'].mean()
        turn_sems = cdf.groupby('turn')['action_value'].sem()

        ax.plot(turn_means.index, turn_means.values,
                color=CONDITION_COLORS[cond],
                label=CONDITION_LABELS[cond],
                linewidth=2, marker='o', markersize=5)
        ax.fill_between(turn_means.index,
                        turn_means.values - 1.96 * turn_sems.values,
                        turn_means.values + 1.96 * turn_sems.values,
                        color=CONDITION_COLORS[cond], alpha=0.15)

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Mean Action Value')
    ax.set_title('Figure 11. Early-Game Trajectories (Turns 1-5, All Games Active)')
    ax.legend(fontsize=10)
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig11_early_game_trajectories.png'))
    plt.savefig(os.path.join(output_dir, 'fig11_early_game_trajectories.pdf'))
    plt.close()
    print("  Saved fig11_early_game_trajectories")


def generate_slope_figure(slopes_df, output_dir):
    """Figure: Within-game escalation slopes by condition."""
    fig, ax = plt.subplots(figsize=(10, 6))
    conditions = ['A', 'B', 'C', 'D']

    for i, cond in enumerate(conditions):
        vals = slopes_df[slopes_df['condition'] == cond]['slope'].values
        ci = bootstrap_ci(vals)

        ax.bar(i, np.mean(vals), color=CONDITION_COLORS[cond], edgecolor='black', alpha=0.8)
        ax.errorbar(i, np.mean(vals),
                     yerr=[[np.mean(vals) - ci[0]], [ci[1] - np.mean(vals)]],
                     fmt='none', color='black', capsize=5, linewidth=1.5)

        # Add individual points
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   c=CONDITION_COLORS[cond], alpha=0.4, s=20, edgecolors='black', linewidth=0.3)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=10)
    ax.set_ylabel('Mean Escalation Slope (per turn)')
    ax.set_title('Figure 12. Within-Game Escalation Trajectory Slopes')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig12_trajectory_slopes.png'))
    plt.savefig(os.path.join(output_dir, 'fig12_trajectory_slopes.pdf'))
    plt.close()
    print("  Saved fig12_trajectory_slopes")


def run_all_round2_analyses(results_dir, figures_dir):
    """Run all Round 2 analyses."""
    np.random.seed(42)

    print("Loading data...")
    df = load_all_results(results_dir)
    game_df = load_game_summaries(results_dir)
    print(f"Loaded {len(game_df)} games, {len(df)} turn-agent observations")

    all_results = {}

    # 1. First-turn analysis
    all_results['first_turn'] = first_turn_analysis(df)

    # 2. Early-game analysis
    all_results['early_game'] = early_game_analysis(df)

    # 3. Escalation rate analysis (game-length controlled)
    all_results['escalation_rate'] = escalation_rate_analysis(df, game_df)

    # 4. Trajectory slopes
    slope_results, slopes_df = trajectory_slope_analysis(df)
    all_results['trajectory_slopes'] = slope_results

    # 5. Detailed breakdown
    all_results['detailed_breakdown'] = detailed_breakdown_table(game_df, df)

    # 6. Cliff's delta
    all_results['cliffs_delta'] = cliffs_delta_all_pairs(game_df)

    # Generate figures
    print("\n" + "="*70)
    print("Generating Round 2 Figures")
    print("="*70)
    generate_first_turn_figure(df, figures_dir)
    generate_early_game_figure(df, figures_dir)
    generate_slope_figure(slopes_df, figures_dir)

    # Save results
    output_path = os.path.join(results_dir, 'round2_analyses.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll Round 2 results saved to {output_path}")

    return all_results


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
    run_all_round2_analyses(results_dir, figures_dir)
