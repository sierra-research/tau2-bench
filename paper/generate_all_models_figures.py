#!/usr/bin/env python3
"""Generate A² score visualizations for ALL models: Grok, DeepSeek, Nemotron, GPT-OSS, MIMO."""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from math import pi
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

# =============================================================================
# Aggregate Data from ALL Sources
# =============================================================================

all_model_data = defaultdict(lambda: {
    'scores': [],
    'by_domain': defaultdict(list),
    'by_strategy': defaultdict(list),
    'by_sophistication': defaultdict(list)
})

# Model display names
model_display = {
    'grok-4.1-fast': 'Grok 4.1 Fast',
    'gpt-oss-120b': 'GPT-OSS-120B',
    'deepseek-v3': 'DeepSeek V3',
    'nemotron-3-nano-30b': 'Nemotron 3 Nano',
    'mimo-v2-flash': 'MIMO V2 Flash',
    'kimi-k2.5': 'Kimi K2.5'
}

# 1. Load from tau2 openrouter_eval raw results
raw_results_dir = Path('../tau2-bench/results/openrouter_eval')
for raw_file in raw_results_dir.glob('raw_results_*.json'):
    try:
        with open(raw_file) as f:
            results = json.load(f)
        for r in results:
            model = r.get('model', '')
            if not model:
                continue
            score = r.get('safety_score', r.get('a2_score', None))
            if score is not None:
                all_model_data[model]['scores'].append(score)

                domain = r.get('domain', '')
                if domain:
                    all_model_data[model]['by_domain'][domain].append(score)

                strategy = r.get('strategy', '')
                if strategy:
                    all_model_data[model]['by_strategy'][strategy].append(score)

                soph = r.get('sophistication', '')
                if soph != '':
                    all_model_data[model]['by_sophistication'][str(soph)].append(score)
    except Exception as e:
        print(f"Error loading {raw_file}: {e}")

# 2. Load from comprehensive results (for DeepSeek)
comprehensive_dir = Path('../experiments/experiments/results/comprehensive')
for comp_file in comprehensive_dir.glob('comprehensive_results_*.json'):
    try:
        with open(comp_file) as f:
            data = json.load(f)

        results = data.get('results', {})
        for model, domains in results.items():
            for domain, domain_data in domains.items():
                seeds = domain_data.get('seeds', [])
                for seed_data in seeds:
                    # Baseline scores
                    baseline = seed_data.get('baseline', {})
                    scores = baseline.get('scores', {})
                    a2 = scores.get('a2', {}).get('mean')
                    if a2 is not None:
                        all_model_data[model]['scores'].append(a2)
                        all_model_data[model]['by_domain'][domain].append(a2)

                    # Adversarial results
                    adversarial = seed_data.get('adversarial', [])
                    for adv in adversarial:
                        strategy = adv.get('strategy', '')
                        soph = adv.get('sophistication', '')
                        adv_results = adv.get('results', {}).get('results', [])
                        for ar in adv_results:
                            result = ar.get('result', {})
                            eval_result = result.get('evaluation', {})
                            a2_score = eval_result.get('a2_score')
                            if a2_score is not None:
                                all_model_data[model]['scores'].append(a2_score)
                                all_model_data[model]['by_domain'][domain].append(a2_score)
                                if strategy:
                                    all_model_data[model]['by_strategy'][strategy].append(a2_score)
                                if soph != '':
                                    all_model_data[model]['by_sophistication'][str(soph)].append(a2_score)
    except Exception as e:
        print(f"Error loading {comp_file}: {e}")

# 3. Load from tau2 metrics files
for metrics_file in raw_results_dir.glob('metrics_*.json'):
    try:
        with open(metrics_file) as f:
            metrics = json.load(f)

        # Extract by_model_domain data
        by_model_domain = metrics.get('by_model_domain', {})
        for model, domains in by_model_domain.items():
            for domain, ddata in domains.items():
                if ddata.get('count', 0) > 0:
                    score = ddata.get('mean_safety_score')
                    if score is not None:
                        # Add weighted by count
                        count = ddata['count']
                        all_model_data[model]['by_domain'][domain].extend([score] * count)

        # Extract by_model_strategy data
        by_model_strategy = metrics.get('by_model_strategy', {})
        for model, strategies in by_model_strategy.items():
            for strategy, sdata in strategies.items():
                if sdata.get('count', 0) > 0:
                    score = sdata.get('mean_safety_score')
                    if score is not None:
                        count = sdata['count']
                        all_model_data[model]['by_strategy'][strategy].extend([score] * count)
    except Exception as e:
        print(f"Error loading {metrics_file}: {e}")

# =============================================================================
# Compute Aggregated Statistics
# =============================================================================

print("="*70)
print("MODELS FOUND AND THEIR DATA:")
print("="*70)

model_stats = {}
for model, data in all_model_data.items():
    if not data['scores'] and not data['by_domain']:
        continue

    # Compute overall mean from all available scores
    all_scores = data['scores'].copy()
    for domain_scores in data['by_domain'].values():
        all_scores.extend(domain_scores)

    if all_scores:
        mean_score = np.mean(all_scores)
        model_stats[model] = {
            'mean': mean_score,
            'count': len(all_scores),
            'by_domain': {d: np.mean(s) for d, s in data['by_domain'].items() if s},
            'by_strategy': {st: np.mean(s) for st, s in data['by_strategy'].items() if s},
            'by_sophistication': {sp: np.mean(s) for sp, s in data['by_sophistication'].items() if s}
        }
        print(f"\n{model_display.get(model, model)}:")
        print(f"  Overall: {mean_score*100:.1f}% (n={len(all_scores)})")
        print(f"  Domains: {model_stats[model]['by_domain']}")

# Filter to models with enough data
valid_models = {m: s for m, s in model_stats.items() if s['count'] >= 5}
print(f"\nValid models (n>=5): {list(valid_models.keys())}")

# Sort models by score
sorted_models = sorted(valid_models.keys(), key=lambda m: valid_models[m]['mean'], reverse=True)

# =============================================================================
# Figure 1: Overall A² Scores by Model (ALL MODELS)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

model_names = [model_display.get(m, m) for m in sorted_models]
model_scores = [valid_models[m]['mean'] * 100 for m in sorted_models]

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_models)))
bars = ax.bar(model_names, model_scores, color=colors, edgecolor='black', linewidth=1.5)

for bar, score, model in zip(bars, model_scores, sorted_models):
    height = bar.get_height()
    count = valid_models[model]['count']
    ax.annotate(f'{score:.1f}%\n(n={count})',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

if model_scores:
    overall_avg = np.mean(model_scores)
    ax.axhline(y=overall_avg, color='#c0392b', linestyle='--', linewidth=2, label=f'Average ({overall_avg:.1f}%)')

ax.set_ylabel('A² Score (%)', fontsize=14)
ax.set_xlabel('Model', fontsize=14)
ax.set_title('Overall A² Scores by Model', fontsize=16, fontweight='bold')
ax.set_ylim(0, 110)
ax.legend(loc='upper right', fontsize=11)
ax.tick_params(axis='x', rotation=15, labelsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'fig1_overall_a2_scores.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig1_overall_a2_scores.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure 1: Overall A² Scores - DONE")

# =============================================================================
# Figure 2: Model × Domain Grouped Bar Chart
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Get all domains
all_domains = set()
for m in sorted_models:
    all_domains.update(valid_models[m]['by_domain'].keys())
domain_list = sorted(all_domains)
domain_labels = [d.capitalize() for d in domain_list]

x = np.arange(len(domain_list))
width = 0.8 / len(sorted_models)
colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_models)))

for i, model in enumerate(sorted_models):
    model_name = model_display.get(model, model)
    scores = []
    for domain in domain_list:
        if domain in valid_models[model]['by_domain']:
            scores.append(valid_models[model]['by_domain'][domain] * 100)
        else:
            scores.append(0)

    offset = (i - len(sorted_models)/2 + 0.5) * width
    bars = ax.bar(x + offset, scores, width, label=model_name, color=colors[i], edgecolor='black', linewidth=0.8)

ax.set_ylabel('A² Score (%)', fontsize=14)
ax.set_xlabel('Domain', fontsize=14)
ax.set_title('A² Scores: Model × Domain', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(domain_labels, fontsize=12)
ax.legend(loc='upper left', fontsize=10, ncol=2)
ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig(output_dir / 'fig2_per_domain_dimensions.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig2_per_domain_dimensions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2: Per Domain - DONE")

# =============================================================================
# Figure 3: Heatmap - Model × Domain
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Build matrix
matrix_data = []
model_labels_plot = []
for model in sorted_models:
    model_labels_plot.append(model_display.get(model, model))
    row = []
    for domain in domain_list:
        if domain in valid_models[model]['by_domain']:
            row.append(valid_models[model]['by_domain'][domain] * 100)
        else:
            row.append(np.nan)
    matrix_data.append(row)

matrix = np.array(matrix_data)

# Use masked array for NaN values
masked_matrix = np.ma.masked_invalid(matrix)
im = ax.imshow(masked_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('A² Score (%)', rotation=-90, va="bottom", fontsize=12)

ax.set_xticks(np.arange(len(domain_list)))
ax.set_yticks(np.arange(len(model_labels_plot)))
ax.set_xticklabels(domain_labels, fontsize=12)
ax.set_yticklabels(model_labels_plot, fontsize=11)

# Add text annotations
for i in range(len(model_labels_plot)):
    for j in range(len(domain_list)):
        if not np.isnan(matrix[i, j]):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}%',
                           ha="center", va="center", color="black", fontsize=10, fontweight='bold')
        else:
            text = ax.text(j, i, 'N/A',
                           ha="center", va="center", color="gray", fontsize=9)

ax.set_title('A² Scores: Model × Domain Heatmap', fontsize=16, fontweight='bold')
ax.set_xlabel('Domain', fontsize=14)
ax.set_ylabel('Model', fontsize=14)

plt.tight_layout()
plt.savefig(output_dir / 'fig3_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig3_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3: Heatmap - DONE")

# =============================================================================
# Figure 4: Radar Chart - Model Profiles by Attack Strategy
# =============================================================================
# Get all strategies
all_strategies = set()
for m in sorted_models:
    all_strategies.update(valid_models[m]['by_strategy'].keys())
strategy_list = sorted(all_strategies)

if strategy_list:
    strategy_labels = [s.replace('_', '\n').title() for s in strategy_list]
    N = len(strategy_list)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_models)))

    for i, model in enumerate(sorted_models):
        model_name = model_display.get(model, model)
        values = []
        for strategy in strategy_list:
            if strategy in valid_models[model]['by_strategy']:
                values.append(valid_models[model]['by_strategy'][strategy] * 100)
            else:
                values.append(0)
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(strategy_labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title('Model A² Profiles by Attack Strategy', fontsize=16, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_radar.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 4: Radar - DONE")

# =============================================================================
# Figure 5: Attack Strategy Vulnerability (Horizontal Bar)
# =============================================================================
if strategy_list:
    fig, ax = plt.subplots(figsize=(12, 7))

    # Average across all models for each strategy
    strategy_avg = {}
    for strategy in strategy_list:
        scores = []
        for model in sorted_models:
            if strategy in valid_models[model]['by_strategy']:
                scores.append(valid_models[model]['by_strategy'][strategy])
        if scores:
            strategy_avg[strategy] = np.mean(scores) * 100

    sorted_strategies = sorted(strategy_avg.keys(), key=lambda s: strategy_avg[s])
    sorted_labels = [s.replace('_', ' ').title() for s in sorted_strategies]
    sorted_scores = [strategy_avg[s] for s in sorted_strategies]

    colors = plt.cm.RdYlGn(np.array(sorted_scores) / 100)
    bars = ax.barh(sorted_labels, sorted_scores, color=colors, edgecolor='black', linewidth=1.2)

    for bar, score in zip(bars, sorted_scores):
        width = bar.get_width()
        ax.annotate(f'{score:.1f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=12, fontweight='bold')

    ax.set_xlabel('A² Score (%) - Higher is Safer', fontsize=14)
    ax.set_ylabel('Attack Strategy', fontsize=14)
    ax.set_title('A² Score by Attack Strategy (Avg Across Models)', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.axvline(x=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_attack_success.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_attack_success.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 5: Attack Strategy - DONE")

# =============================================================================
# Figure 6: Model × Strategy Heatmap
# =============================================================================
if strategy_list:
    fig, ax = plt.subplots(figsize=(14, 8))

    matrix_data = []
    for model in sorted_models:
        row = []
        for strategy in strategy_list:
            if strategy in valid_models[model]['by_strategy']:
                row.append(valid_models[model]['by_strategy'][strategy] * 100)
            else:
                row.append(np.nan)
        matrix_data.append(row)

    matrix = np.array(matrix_data)
    masked_matrix = np.ma.masked_invalid(matrix)

    im = ax.imshow(masked_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('A² Score (%)', rotation=-90, va="bottom", fontsize=12)

    strategy_labels_short = [s.replace('_', ' ').title()[:15] for s in strategy_list]
    ax.set_xticks(np.arange(len(strategy_list)))
    ax.set_yticks(np.arange(len(sorted_models)))
    ax.set_xticklabels(strategy_labels_short, fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels([model_display.get(m, m) for m in sorted_models], fontsize=11)

    for i in range(len(sorted_models)):
        for j in range(len(strategy_list)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.0f}%', ha="center", va="center",
                       color="black", fontsize=9, fontweight='bold')

    ax.set_title('A² Scores: Model × Attack Strategy', fontsize=16, fontweight='bold')
    ax.set_xlabel('Attack Strategy', fontsize=14)
    ax.set_ylabel('Model', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_model_domain_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig6_model_domain_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 6: Model×Strategy Heatmap - DONE")

# =============================================================================
# Print Summary Report
# =============================================================================
print("\n" + "="*70)
print("A² SCORE SUMMARY REPORT - ALL MODELS")
print("="*70)

print("\n--- Overall A² Scores (Ranked) ---")
for model in sorted_models:
    stats = valid_models[model]
    name = model_display.get(model, model)
    print(f"  {name:<20}: {stats['mean']*100:>6.1f}% (n={stats['count']})")

print("\n--- By Domain ---")
for domain in domain_list:
    print(f"\n  {domain.upper()}:")
    for model in sorted_models:
        if domain in valid_models[model]['by_domain']:
            score = valid_models[model]['by_domain'][domain] * 100
            name = model_display.get(model, model)
            print(f"    {name:<20}: {score:>6.1f}%")

print("\n--- Model × Domain Matrix ---")
header = f"{'Model':<20}"
for d in domain_list:
    header += f" | {d.capitalize():>10}"
print(header)
print("-" * (22 + 13 * len(domain_list)))
for model in sorted_models:
    row = f"{model_display.get(model, model):<20}"
    for domain in domain_list:
        if domain in valid_models[model]['by_domain']:
            score = valid_models[model]['by_domain'][domain] * 100
            row += f" | {score:>9.1f}%"
        else:
            row += f" | {'N/A':>10}"
    print(row)

print("\n" + "="*70)
print(f"All figures saved to: {output_dir.absolute()}")
print("="*70)
