"""
Additional analyses to address reviewer feedback.

R1: Multiple comparison corrections (Bonferroni, Holm)
R2: Game-level effect sizes (not turn-level)
R3: Condition x model interaction test
R4: Bootstrap confidence intervals
R5: Bootstrap significance tests
R10: Per-model condition effects
R11: Scenario effects analysis
R7: Qualitative reasoning examples
R9: Document temperature/sampling parameters
R14: Scenario x condition interaction
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment.analysis import load_all_results, load_game_summaries, count_evaluative_dimensions

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CONDITION_COLORS = {'A': '#d62728', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#1f77b4'}
CONDITION_LABELS = {'A': 'A: Control', 'B': 'B: Neutralized', 'C': 'C: Mutual Rat.', 'D': 'D: Op. Constraint'}
MODEL_LABELS = {'clemson-qwen3-30b': 'Qwen3 30B', 'cf-llama-70b': 'Llama 3.3 70B'}


def bootstrap_ci(data, n_boot=10000, ci=95, statistic=np.mean):
    """Compute bootstrap confidence interval."""
    data = np.array(data)
    if len(data) == 0:
        return (np.nan, np.nan)
    boot_stats = np.array([statistic(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boot_stats, (100 - ci) / 2)
    hi = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return (lo, hi)


def permutation_test(group1, group2, n_perm=10000, statistic='mean_diff'):
    """Two-sample permutation test."""
    group1 = np.array(group1)
    group2 = np.array(group2)
    observed = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(combined)
        perm_diff = np.mean(perm[:n1]) - np.mean(perm[n1:])
        if abs(perm_diff) >= abs(observed):
            count += 1
    return count / n_perm


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction to a list of (label, p_value) tuples."""
    sorted_pvals = sorted(p_values, key=lambda x: x[1])
    m = len(sorted_pvals)
    corrected = []
    for i, (label, p) in enumerate(sorted_pvals):
        adj_p = min(p * (m - i), 1.0)
        corrected.append((label, p, adj_p))
    return corrected


def game_level_cohens_d(game_df, cond1, cond2, metric='mean_escalation'):
    """Compute Cohen's d at game level (independent observations)."""
    g1 = game_df[game_df['condition'] == cond1][metric].values
    g2 = game_df[game_df['condition'] == cond2][metric].values
    if len(g1) == 0 or len(g2) == 0:
        return np.nan
    pooled_std = np.sqrt(((len(g1) - 1) * np.var(g1, ddof=1) + (len(g2) - 1) * np.var(g2, ddof=1)) / (len(g1) + len(g2) - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(g1) - np.mean(g2)) / pooled_std


def aligned_rank_transform_interaction(game_df):
    """Test condition x model interaction using aligned rank transform.
    Approximation using two-way ANOVA on ranked data."""
    from itertools import product

    df = game_df[['condition', 'model', 'max_escalation']].copy()
    # Compute cell means and marginal means
    grand_mean = df['max_escalation'].mean()
    cond_means = df.groupby('condition')['max_escalation'].transform('mean')
    model_means = df.groupby('model')['max_escalation'].transform('mean')

    # Aligned for interaction: Y - cond_effect - model_effect + grand_mean
    df['aligned_interaction'] = df['max_escalation'] - cond_means - model_means + grand_mean
    df['ranked_interaction'] = stats.rankdata(df['aligned_interaction'])

    # Now run ANOVA on ranked aligned data for interaction
    cells = defaultdict(list)
    for _, row in df.iterrows():
        cells[(row['condition'], row['model'])].append(row['ranked_interaction'])

    # F-test for interaction
    conditions = sorted(df['condition'].unique())
    models = sorted(df['model'].unique())

    # Simple two-way ANOVA by hand for interaction term
    n_total = len(df)
    ss_total = np.sum((df['ranked_interaction'] - df['ranked_interaction'].mean()) ** 2)

    # SS for condition main effect on interaction-aligned ranks
    ss_cond = sum(
        len(df[df['condition'] == c]) * (df[df['condition'] == c]['ranked_interaction'].mean() - df['ranked_interaction'].mean()) ** 2
        for c in conditions
    )
    # SS for model main effect on interaction-aligned ranks
    ss_model = sum(
        len(df[df['model'] == m]) * (df[df['model'] == m]['ranked_interaction'].mean() - df['ranked_interaction'].mean()) ** 2
        for m in models
    )
    # SS for interaction
    ss_cells = sum(
        len(v) * (np.mean(v) - df['ranked_interaction'].mean()) ** 2
        for v in cells.values()
    )
    ss_interaction = ss_cells - ss_cond - ss_model

    df_interaction = (len(conditions) - 1) * (len(models) - 1)
    df_residual = n_total - len(conditions) * len(models)
    ss_residual = ss_total - ss_cells

    if df_residual > 0 and ss_residual > 0:
        ms_interaction = ss_interaction / df_interaction
        ms_residual = ss_residual / df_residual
        f_stat = ms_interaction / ms_residual
        p_value = 1 - stats.f.cdf(f_stat, df_interaction, df_residual)
    else:
        f_stat, p_value = np.nan, np.nan

    return {
        'F_interaction': f_stat,
        'p_interaction': p_value,
        'df_interaction': df_interaction,
        'df_residual': df_residual,
        'ss_interaction': ss_interaction,
        'ss_residual': ss_residual,
    }


def fig_per_model_condition(game_df, df, output_dir):
    """Figure: Condition effects within each model separately (R10)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    models = sorted(game_df['model'].unique())
    conditions = ['A', 'B', 'C', 'D']

    for row_idx, model in enumerate(models):
        mdf_game = game_df[game_df['model'] == model]
        mdf_turn = df[df['model'] == model]

        # Panel 1: Mean escalation value
        ax = axes[row_idx][0]
        means, cis_lo, cis_hi = [], [], []
        for cond in conditions:
            vals = mdf_turn[mdf_turn['condition'] == cond]['action_value'].values
            m = np.mean(vals) if len(vals) > 0 else 0
            ci = bootstrap_ci(vals)
            means.append(m)
            cis_lo.append(ci[0])
            cis_hi.append(ci[1])
        bars = ax.bar(range(4), means, color=[CONDITION_COLORS[c] for c in conditions], edgecolor='black')
        ax.errorbar(range(4), means, yerr=[np.array(means) - np.array(cis_lo), np.array(cis_hi) - np.array(means)],
                     fmt='none', color='black', capsize=5)
        ax.set_xticks(range(4))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=8)
        ax.set_ylabel('Mean Action Value')
        ax.set_title(f'{MODEL_LABELS.get(model, model)}: Mean Escalation')

        # Panel 2: De-escalation rate
        ax = axes[row_idx][1]
        rates = []
        for cond in conditions:
            cdf = mdf_turn[mdf_turn['condition'] == cond]
            rate = cdf['is_deescalatory'].mean() * 100 if len(cdf) > 0 else 0
            rates.append(rate)
        bars = ax.bar(range(4), rates, color=[CONDITION_COLORS[c] for c in conditions], edgecolor='black')
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f'{r:.0f}%', ha='center', fontsize=9)
        ax.set_xticks(range(4))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=8)
        ax.set_ylabel('De-escalation Rate (%)')
        ax.set_title(f'{MODEL_LABELS.get(model, model)}: De-escalation Rate')

        # Panel 3: Max escalation boxplot
        ax = axes[row_idx][2]
        data = [mdf_game[mdf_game['condition'] == c]['max_escalation'].values for c in conditions]
        bp = ax.boxplot(data, labels=[CONDITION_LABELS[c] for c in conditions],
                        patch_artist=True, widths=0.5)
        for patch, cond in zip(bp['boxes'], conditions):
            patch.set_facecolor(CONDITION_COLORS[cond])
            patch.set_alpha(0.6)
        ax.axhline(y=550, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Max Escalation')
        ax.set_title(f'{MODEL_LABELS.get(model, model)}: Max Escalation')
        for label in ax.get_xticklabels():
            label.set_fontsize(8)

    fig.suptitle('Figure 8. Condition Effects Within Each Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig8_per_model_condition.png'))
    plt.savefig(os.path.join(output_dir, 'fig8_per_model_condition.pdf'))
    plt.close()
    print("  Saved fig8_per_model_condition")


def fig_scenario_effects(game_df, df, output_dir):
    """Figure: Scenario effects (R11)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    conditions = ['A', 'B', 'C', 'D']
    scenarios = sorted(game_df['scenario'].unique())

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        sdf = game_df[game_df['scenario'] == scenario]
        means = [sdf[sdf['condition'] == c]['mean_escalation'].mean() for c in conditions]
        bars = ax.bar(range(4), means, color=[CONDITION_COLORS[c] for c in conditions], edgecolor='black')
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, f'{m:.0f}', ha='center', fontsize=10)
        ax.set_xticks(range(4))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=9)
        ax.set_ylabel('Mean Escalation Value')
        ax.set_title(scenario.replace('_', ' ').title())
        ax.set_ylim(0, max(means) * 1.3 if means else 100)

    fig.suptitle('Figure 9. Condition Effects by Scenario', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig9_scenario_effects.png'))
    plt.savefig(os.path.join(output_dir, 'fig9_scenario_effects.pdf'))
    plt.close()
    print("  Saved fig9_scenario_effects")


def extract_reasoning_examples(df, output_path):
    """Extract representative reasoning examples across conditions (R7)."""
    examples = {}
    np.random.seed(42)

    for cond in ['A', 'B', 'C', 'D']:
        cdf = df[(df['condition'] == cond) & (df['reasoning'].str.len() > 50)]
        if len(cdf) == 0:
            continue

        # Get one early-game and one mid-game example
        early = cdf[cdf['turn'] <= 3]
        mid = cdf[(cdf['turn'] >= 5) & (cdf['turn'] <= 10)]

        examples[cond] = {
            'early': early.sample(1).iloc[0] if len(early) > 0 else None,
            'mid': mid.sample(1).iloc[0] if len(mid) > 0 else None,
        }

    # Write examples to file
    with open(output_path, 'w') as f:
        for cond in ['A', 'B', 'C', 'D']:
            if cond not in examples:
                continue
            f.write(f"\n{'='*80}\n")
            f.write(f"CONDITION {cond}\n")
            f.write(f"{'='*80}\n")
            for phase in ['early', 'mid']:
                row = examples[cond][phase]
                if row is None:
                    continue
                f.write(f"\n--- {phase.title()} game (Turn {row['turn']}, {row['agent']}, "
                        f"model={row['model']}, scenario={row['scenario']}) ---\n")
                f.write(f"Action: {row['action_code']} (value={row['action_value']})\n")
                f.write(f"Signal: {row['signal_code']}\n")
                f.write(f"Reasoning:\n{row['reasoning'][:1500]}\n")
                dims = count_evaluative_dimensions(row['reasoning'])
                f.write(f"Dimensional density: {dims}\n")

    print(f"  Saved reasoning examples to {output_path}")
    return examples


def run_all_reviewer_analyses(results_dir, figures_dir):
    """Run all additional analyses requested by reviewers."""
    np.random.seed(42)

    print("Loading data...")
    df = load_all_results(results_dir)
    game_df = load_game_summaries(results_dir)
    print(f"Loaded {len(game_df)} games, {len(df)} turn-agent observations\n")

    all_results = {}

    # ═══════════════════════════════════════════════════════════════════════
    # R1: Multiple comparison corrections
    # ═══════════════════════════════════════════════════════════════════════
    print("R1: Multiple comparison corrections")
    pairwise_pvals = []
    for cond in ['B', 'C', 'D']:
        a_data = game_df[game_df['condition'] == 'A']['max_escalation'].values
        c_data = game_df[game_df['condition'] == cond]['max_escalation'].values
        _, p = stats.mannwhitneyu(a_data, c_data, alternative='two-sided')
        pairwise_pvals.append((f'A_vs_{cond}', p))

    # Also B vs D, C vs D, B vs C
    for c1, c2 in [('B', 'C'), ('B', 'D'), ('C', 'D')]:
        d1 = game_df[game_df['condition'] == c1]['max_escalation'].values
        d2 = game_df[game_df['condition'] == c2]['max_escalation'].values
        _, p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
        pairwise_pvals.append((f'{c1}_vs_{c2}', p))

    holm_results = holm_bonferroni(pairwise_pvals)
    bonferroni_results = [(label, p, min(p * len(pairwise_pvals), 1.0)) for label, p in pairwise_pvals]

    print("  Holm-Bonferroni corrected p-values:")
    for label, raw_p, adj_p in holm_results:
        print(f"    {label}: raw p={raw_p:.4f}, Holm-adjusted p={adj_p:.4f}")

    all_results['holm_corrected'] = {label: {'raw_p': raw_p, 'adjusted_p': adj_p} for label, raw_p, adj_p in holm_results}
    all_results['bonferroni_corrected'] = {label: {'raw_p': raw_p, 'adjusted_p': adj_p} for label, raw_p, adj_p in bonferroni_results}

    # ═══════════════════════════════════════════════════════════════════════
    # R2: Game-level effect sizes
    # ═══════════════════════════════════════════════════════════════════════
    print("\nR2: Game-level Cohen's d (mean_escalation)")
    game_d = {}
    for cond in ['B', 'C', 'D']:
        d_val = game_level_cohens_d(game_df, 'A', cond, 'mean_escalation')
        game_d[f'A_vs_{cond}'] = d_val
        print(f"  A vs {cond}: d={d_val:.3f}")
    all_results['game_level_cohens_d_mean_esc'] = game_d

    print("  Game-level Cohen's d (max_escalation)")
    game_d_max = {}
    for cond in ['B', 'C', 'D']:
        d_val = game_level_cohens_d(game_df, 'A', cond, 'max_escalation')
        game_d_max[f'A_vs_{cond}'] = d_val
        print(f"  A vs {cond}: d={d_val:.3f}")
    all_results['game_level_cohens_d_max_esc'] = game_d_max

    # ═══════════════════════════════════════════════════════════════════════
    # R3: Condition x model interaction
    # ═══════════════════════════════════════════════════════════════════════
    print("\nR3: Condition x Model interaction (Aligned Rank Transform)")
    interaction = aligned_rank_transform_interaction(game_df)
    print(f"  F({interaction['df_interaction']},{interaction['df_residual']}) = {interaction['F_interaction']:.3f}, "
          f"p = {interaction['p_interaction']:.4f}")
    all_results['condition_x_model_interaction'] = interaction

    # Also: simple two-way ANOVA
    # Main effects using rank-based approach
    print("\n  Two-way ANOVA on max_escalation:")
    from scipy.stats import f_oneway
    grand_mean = game_df['max_escalation'].mean()
    ss_total = ((game_df['max_escalation'] - grand_mean) ** 2).sum()

    conditions = sorted(game_df['condition'].unique())
    models = sorted(game_df['model'].unique())

    ss_cond = sum(
        len(game_df[game_df['condition'] == c]) *
        (game_df[game_df['condition'] == c]['max_escalation'].mean() - grand_mean) ** 2
        for c in conditions
    )
    ss_model = sum(
        len(game_df[game_df['model'] == m]) *
        (game_df[game_df['model'] == m]['max_escalation'].mean() - grand_mean) ** 2
        for m in models
    )

    # Cell-level SS
    ss_cells = 0
    for c in conditions:
        for m in models:
            cell = game_df[(game_df['condition'] == c) & (game_df['model'] == m)]
            if len(cell) > 0:
                ss_cells += len(cell) * (cell['max_escalation'].mean() - grand_mean) ** 2
    ss_int = ss_cells - ss_cond - ss_model
    ss_resid = ss_total - ss_cells

    n_total = len(game_df)
    df_cond = len(conditions) - 1
    df_model = len(models) - 1
    df_int = df_cond * df_model
    df_resid = n_total - len(conditions) * len(models)

    results_anova = {}
    for name, ss, dof in [('condition', ss_cond, df_cond), ('model', ss_model, df_model), ('interaction', ss_int, df_int)]:
        ms = ss / dof if dof > 0 else 0
        ms_resid = ss_resid / df_resid if df_resid > 0 else 1
        f_val = ms / ms_resid if ms_resid > 0 else 0
        p_val = 1 - stats.f.cdf(f_val, dof, df_resid) if dof > 0 and df_resid > 0 else np.nan
        eta_sq = ss / ss_total if ss_total > 0 else 0
        results_anova[name] = {'SS': ss, 'df': dof, 'F': f_val, 'p': p_val, 'eta_squared': eta_sq}
        print(f"    {name}: SS={ss:.1f}, F({dof},{df_resid})={f_val:.3f}, p={p_val:.4f}, η²={eta_sq:.3f}")

    all_results['two_way_anova'] = results_anova

    # ═══════════════════════════════════════════════════════════════════════
    # R4 & R5: Bootstrap CIs and permutation tests
    # ═══════════════════════════════════════════════════════════════════════
    print("\nR4/R5: Bootstrap CIs and permutation tests")
    boot_results = {}
    for cond in ['A', 'B', 'C', 'D']:
        vals = game_df[game_df['condition'] == cond]['mean_escalation'].values
        ci = bootstrap_ci(vals, n_boot=10000)
        boot_results[cond] = {
            'mean': float(np.mean(vals)),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n': len(vals)
        }
        print(f"  Cond {cond}: mean={np.mean(vals):.1f}, 95% CI=[{ci[0]:.1f}, {ci[1]:.1f}]")

    all_results['bootstrap_ci_mean_escalation'] = boot_results

    # Permutation tests for A vs each
    print("\n  Permutation tests (A vs each, 10000 permutations):")
    perm_results = {}
    a_vals = game_df[game_df['condition'] == 'A']['mean_escalation'].values
    for cond in ['B', 'C', 'D']:
        c_vals = game_df[game_df['condition'] == cond]['mean_escalation'].values
        p_perm = permutation_test(a_vals, c_vals, n_perm=10000)
        perm_results[f'A_vs_{cond}'] = p_perm
        print(f"    A vs {cond}: permutation p={p_perm:.4f}")

    all_results['permutation_tests'] = perm_results

    # Bootstrap CIs for de-escalation rates
    print("\n  Bootstrap CIs for de-escalation rates:")
    deesc_boot = {}
    for cond in ['A', 'B', 'C', 'D']:
        cdf = df[df['condition'] == cond]
        # Bootstrap on game-level de-escalation rates
        game_rates = []
        for gid in cdf['game_id'].unique():
            gdf = cdf[cdf['game_id'] == gid]
            rate = gdf['is_deescalatory'].mean()
            game_rates.append(rate)
        game_rates = np.array(game_rates)
        ci = bootstrap_ci(game_rates * 100, n_boot=10000)
        deesc_boot[cond] = {
            'mean': float(np.mean(game_rates) * 100),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
        }
        print(f"  Cond {cond}: de-esc rate={np.mean(game_rates)*100:.1f}%, 95% CI=[{ci[0]:.1f}%, {ci[1]:.1f}%]")

    all_results['bootstrap_ci_deescalation'] = deesc_boot

    # ═══════════════════════════════════════════════════════════════════════
    # R10: Per-model breakdown
    # ═══════════════════════════════════════════════════════════════════════
    print("\nR10: Per-model condition effects")
    for model in sorted(game_df['model'].unique()):
        print(f"\n  Model: {MODEL_LABELS.get(model, model)}")
        mdf = game_df[game_df['model'] == model]
        for cond in ['A', 'B', 'C', 'D']:
            cdf = mdf[mdf['condition'] == cond]
            if len(cdf) > 0:
                print(f"    Cond {cond}: n={len(cdf)}, mean_max_esc={cdf['max_escalation'].mean():.0f}, "
                      f"mean_esc={cdf['mean_escalation'].mean():.0f}, "
                      f"nuclear={cdf['nuclear_used'].mean()*100:.0f}%, "
                      f"deesc_total={cdf['deesc_count'].sum()}")

    # Per-model Mann-Whitney A vs D
    print("\n  Per-model Mann-Whitney A vs D:")
    per_model_tests = {}
    for model in sorted(game_df['model'].unique()):
        mdf = game_df[game_df['model'] == model]
        a = mdf[mdf['condition'] == 'A']['mean_escalation'].values
        d = mdf[mdf['condition'] == 'D']['mean_escalation'].values
        if len(a) > 1 and len(d) > 1:
            stat, p = stats.mannwhitneyu(a, d, alternative='two-sided')
            d_val = game_level_cohens_d(mdf, 'A', 'D', 'mean_escalation')
            per_model_tests[model] = {'U': stat, 'p': p, 'd': d_val}
            print(f"    {MODEL_LABELS.get(model, model)}: U={stat:.1f}, p={p:.4f}, d={d_val:.3f}")

    all_results['per_model_A_vs_D'] = per_model_tests

    # ═══════════════════════════════════════════════════════════════════════
    # R11: Scenario effects
    # ═══════════════════════════════════════════════════════════════════════
    print("\nR11: Scenario effects")
    for scenario in sorted(game_df['scenario'].unique()):
        sdf = game_df[game_df['scenario'] == scenario]
        print(f"\n  Scenario: {scenario}")
        for cond in ['A', 'B', 'C', 'D']:
            cdf = sdf[sdf['condition'] == cond]
            if len(cdf) > 0:
                print(f"    Cond {cond}: n={len(cdf)}, mean_max_esc={cdf['max_escalation'].mean():.0f}, "
                      f"mean_esc={cdf['mean_escalation'].mean():.0f}")

    # Scenario main effect
    for scenario in sorted(game_df['scenario'].unique()):
        sdf = game_df[game_df['scenario'] == scenario]
        a = sdf[sdf['condition'] == 'A']['mean_escalation'].values
        d = sdf[sdf['condition'] == 'D']['mean_escalation'].values
        if len(a) > 1 and len(d) > 1:
            stat, p = stats.mannwhitneyu(a, d, alternative='two-sided')
            print(f"  {scenario} A vs D: U={stat:.1f}, p={p:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # R8: Validate dimensional density measure
    # ═══════════════════════════════════════════════════════════════════════
    print("\nR8: Dimensional density validation")
    df_with_dims = df.copy()
    df_with_dims['dim_density'] = df_with_dims['reasoning'].apply(count_evaluative_dimensions)

    # Statistical test on dimensional density
    groups = [df_with_dims[df_with_dims['condition'] == c]['dim_density'].values for c in ['A', 'B', 'C', 'D']]
    h_stat, h_p = stats.kruskal(*[g for g in groups if len(g) > 0])
    print(f"  Kruskal-Wallis on dim density: H={h_stat:.3f}, p={h_p:.4f}")

    # Per-condition stats
    for cond in ['A', 'B', 'C', 'D']:
        vals = df_with_dims[df_with_dims['condition'] == cond]['dim_density'].values
        ci = bootstrap_ci(vals, n_boot=10000)
        print(f"  Cond {cond}: mean={np.mean(vals):.2f}, SD={np.std(vals):.2f}, 95% CI=[{ci[0]:.2f}, {ci[1]:.2f}]")

    all_results['dim_density_kruskal'] = {'H': h_stat, 'p': h_p}

    # ═══════════════════════════════════════════════════════════════════════
    # Generate new figures
    # ═══════════════════════════════════════════════════════════════════════
    print("\nGenerating new figures...")
    fig_per_model_condition(game_df, df, figures_dir)
    fig_scenario_effects(game_df, df, figures_dir)

    # ═══════════════════════════════════════════════════════════════════════
    # R7: Extract reasoning examples
    # ═══════════════════════════════════════════════════════════════════════
    print("\nR7: Extracting reasoning examples...")
    examples_path = os.path.join(results_dir, 'reasoning_examples.txt')
    extract_reasoning_examples(df, examples_path)

    # ═══════════════════════════════════════════════════════════════════════
    # Save all results
    # ═══════════════════════════════════════════════════════════════════════
    output_path = os.path.join(results_dir, 'reviewer_analyses.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {output_path}")

    return all_results


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
    run_all_reviewer_analyses(results_dir, figures_dir)
