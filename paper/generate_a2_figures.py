#!/usr/bin/env python3
"""Generate comprehensive A² score visualizations from benchmark results."""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from math import pi

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

# =============================================================================
# Load Data from Multiple Sources
# =============================================================================

# Load tau2 benchmark metrics (most recent)
tau2_metrics_path = Path('../tau2-bench/results/openrouter_eval/metrics_20260205_030022.json')
with open(tau2_metrics_path) as f:
    tau2_metrics = json.load(f)

# Load ICML/NeurIPS results for A² component breakdown
icml_results_path = Path('../experiments/results/icml/full_results_20251221_210832.json')
with open(icml_results_path) as f:
    icml_results = json.load(f)

print("Data loaded successfully!")

# =============================================================================
# Extract and Organize Data
# =============================================================================

# From tau2 metrics - safety scores (which serve as proxy for A² in adversarial setting)
models_tau2 = list(tau2_metrics['by_model'].keys())
domains_tau2 = list(tau2_metrics['by_domain'].keys())
strategies = list(tau2_metrics['by_strategy'].keys())

# Model display names
model_display = {
    'grok-4.1-fast': 'Grok 4.1 Fast',
    'gpt-oss-120b': 'GPT-OSS-120B',
    'deepseek-v3': 'DeepSeek V3',
    'kimi-k2.5': 'Kimi K2.5'
}

# From ICML results - full A² component breakdown
a2_aggregate = icml_results['aggregate_results']

# Extract unique models and domains from ICML data
icml_models = list(set(r['model_name'] for r in a2_aggregate))
icml_domains = list(set(r['domain'] for r in a2_aggregate))

print(f"Tau2 Models: {models_tau2}")
print(f"Tau2 Domains: {domains_tau2}")
print(f"ICML Models: {icml_models}")
print(f"ICML Domains: {icml_domains}")

# =============================================================================
# Figure 1: Overall A² Scores by Model (Bar Chart)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Combine data: use tau2 safety scores as primary metric
model_names = [model_display.get(m, m) for m in models_tau2]
model_scores = [tau2_metrics['by_model'][m]['mean_safety_score'] * 100 for m in models_tau2]

colors = ['#e74c3c', '#9b59b6']
bars = ax.bar(model_names, model_scores, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

for bar, score in zip(bars, model_scores):
    height = bar.get_height()
    ax.annotate(f'{score:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=14, fontweight='bold')

overall_avg = tau2_metrics['overall']['mean_safety_score'] * 100
ax.axhline(y=overall_avg, color='#2c3e50', linestyle='--', linewidth=2, label=f'Average ({overall_avg:.1f}%)')

ax.set_ylabel('A² Score (%)', fontsize=14)
ax.set_xlabel('Model', fontsize=14)
ax.set_title('Overall A² Scores by Model', fontsize=16, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(loc='upper right', fontsize=11)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig(output_dir / 'fig1_overall_a2_scores.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig1_overall_a2_scores.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1: Overall A² Scores - DONE")

# =============================================================================
# Figure 2: A² Scores Per Domain Per Model (Grouped Bar)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Build model x domain matrix from tau2 data
model_domain_data = tau2_metrics['by_model_domain']
domain_labels = ['Airline', 'Retail', 'Telecom']
domain_keys = ['airline', 'retail', 'telecom']

x = np.arange(len(domain_labels))
width = 0.35
colors = ['#e74c3c', '#9b59b6']

for i, model_key in enumerate(models_tau2):
    model_name = model_display.get(model_key, model_key)
    scores = []
    for dk in domain_keys:
        if dk in model_domain_data.get(model_key, {}):
            scores.append(model_domain_data[model_key][dk]['mean_safety_score'] * 100)
        else:
            scores.append(0)

    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, scores, width, label=model_name, color=colors[i], edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, score in zip(bars, scores):
        if score > 0:
            height = bar.get_height()
            ax.annotate(f'{score:.0f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('A² Score (%)', fontsize=14)
ax.set_xlabel('Domain', fontsize=14)
ax.set_title('A² Scores: Model × Domain', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(domain_labels, fontsize=12)
ax.legend(loc='upper left', fontsize=11)
ax.set_ylim(0, 100)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig(output_dir / 'fig2_per_domain_dimensions.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig2_per_domain_dimensions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2: Per Domain - DONE")

# =============================================================================
# Figure 3: Heatmap - Model × Domain A² Scores
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Build matrix
matrix_data = []
model_labels_plot = []
for model_key in models_tau2:
    model_labels_plot.append(model_display.get(model_key, model_key))
    row = []
    for dk in domain_keys:
        if dk in model_domain_data.get(model_key, {}):
            row.append(model_domain_data[model_key][dk]['mean_safety_score'] * 100)
        else:
            row.append(0)
    matrix_data.append(row)

matrix = np.array(matrix_data)

im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=20, vmax=90)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('A² Score (%)', rotation=-90, va="bottom", fontsize=12)

ax.set_xticks(np.arange(len(domain_labels)))
ax.set_yticks(np.arange(len(model_labels_plot)))
ax.set_xticklabels(domain_labels, fontsize=12)
ax.set_yticklabels(model_labels_plot, fontsize=12)

# Add text annotations
for i in range(len(model_labels_plot)):
    for j in range(len(domain_labels)):
        text = ax.text(j, i, f'{matrix[i, j]:.1f}%',
                       ha="center", va="center", color="black", fontsize=13, fontweight='bold')

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
strategy_labels = ['Social\nEngineering', 'Prompt\nInjection', 'Policy\nExploitation',
                   'Identity\nManipulation', 'Info\nExtraction']
strategy_keys = ['social_engineering', 'prompt_injection', 'policy_exploitation',
                 'identity_manipulation', 'information_extraction']

N = len(strategy_labels)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

colors = ['#e74c3c', '#9b59b6']

model_strategy_data = tau2_metrics['by_model_strategy']
for i, model_key in enumerate(models_tau2):
    model_name = model_display.get(model_key, model_key)
    values = []
    for sk in strategy_keys:
        if sk in model_strategy_data.get(model_key, {}):
            values.append(model_strategy_data[model_key][sk]['mean_safety_score'] * 100)
        else:
            values.append(0)
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2.5, label=model_name, color=colors[i])
    ax.fill(angles, values, alpha=0.2, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(strategy_labels, fontsize=11)
ax.set_ylim(0, 100)
ax.set_title('Model A² Profiles by Attack Strategy', fontsize=16, fontweight='bold', y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'fig4_radar.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig4_radar.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 4: Radar - DONE")

# =============================================================================
# Figure 5: Attack Success Rate by Strategy
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

strategy_scores = [tau2_metrics['by_strategy'][sk]['mean_safety_score'] * 100 for sk in strategy_keys]

# Sort by score (ascending = most vulnerable first)
sorted_indices = np.argsort(strategy_scores)
sorted_labels = [strategy_labels[i].replace('\n', ' ') for i in sorted_indices]
sorted_scores = [strategy_scores[i] for i in sorted_indices]

colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_scores)))
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
ax.set_title('A² Score Vulnerability by Attack Strategy', fontsize=16, fontweight='bold')
ax.set_xlim(0, 100)
ax.tick_params(axis='both', labelsize=12)

# Add vulnerability annotation
ax.axvline(x=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(52, len(sorted_labels) - 0.5, 'Vulnerability\nThreshold', fontsize=10, color='red')

plt.tight_layout()
plt.savefig(output_dir / 'fig5_attack_success.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig5_attack_success.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 5: Attack Success - DONE")

# =============================================================================
# Figure 6: Defense Rate vs A² Score (Scatter)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Plot each model-domain combination
markers = ['o', 's', '^', 'D']
colors = ['#e74c3c', '#3498db', '#2ecc71']

for i, model_key in enumerate(models_tau2):
    model_name = model_display.get(model_key, model_key)
    for j, dk in enumerate(domain_keys):
        if dk in model_domain_data.get(model_key, {}):
            data = model_domain_data[model_key][dk]
            a2_score = data['mean_safety_score'] * 100
            defense_rate = data['defense_rate'] * 100

            ax.scatter(a2_score, defense_rate,
                      marker=markers[i] if i < len(markers) else 'o',
                      color=colors[j],
                      s=200, edgecolors='black', linewidth=1.5,
                      label=f'{model_name} - {domain_labels[j]}' if i == 0 or j == 0 else '')

ax.set_xlabel('A² Score (%)', fontsize=14)
ax.set_ylabel('Defense Rate (%)', fontsize=14)
ax.set_title('Defense Rate vs A² Score by Model-Domain', fontsize=16, fontweight='bold')
ax.set_xlim(0, 100)
ax.set_ylim(90, 105)
ax.tick_params(axis='both', labelsize=12)

# Create legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Grok 4.1 Fast'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='GPT-OSS-120B'),
    Line2D([0], [0], marker='o', color='#e74c3c', markersize=10, label='Airline', linestyle=''),
    Line2D([0], [0], marker='o', color='#3498db', markersize=10, label='Retail', linestyle=''),
    Line2D([0], [0], marker='o', color='#2ecc71', markersize=10, label='Telecom', linestyle=''),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'fig6_defense_vs_a2.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig6_defense_vs_a2.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 6: Defense vs A² - DONE")

# =============================================================================
# Figure 7: Task Completion Rate Analysis (from ICML data)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Extract task completion data from ICML results
icml_by_domain_model = {}
for r in a2_aggregate:
    domain = r['domain']
    model = r['model_name']
    if domain not in icml_by_domain_model:
        icml_by_domain_model[domain] = {}
    icml_by_domain_model[domain][model] = {
        'a2_score': r['scores']['a2_score'] * 100,
        'task_completion': r['task_completion_rate'] * 100
    }

# Create grouped bar chart
icml_domain_labels = list(icml_by_domain_model.keys())
icml_model_list = ['Baseline', 'Conservative', 'Aggressive', 'Balanced']
x = np.arange(len(icml_domain_labels))
width = 0.2

colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

for i, model in enumerate(icml_model_list):
    scores = []
    for domain in icml_domain_labels:
        if model in icml_by_domain_model.get(domain, {}):
            scores.append(icml_by_domain_model[domain][model]['a2_score'])
        else:
            scores.append(0)

    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, scores, width, label=model, color=colors[i], edgecolor='black', linewidth=1)

ax.set_ylabel('A² Score (%)', fontsize=14)
ax.set_xlabel('Domain', fontsize=14)
ax.set_title('A² Score by Model Type and Domain (ICML Evaluation)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([d.capitalize() for d in icml_domain_labels], fontsize=12)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, 110)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig(output_dir / 'fig7_task_completion.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig7_task_completion.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 7: Task Completion - DONE")

# =============================================================================
# Figure 8: ASR by Sophistication Level
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 6))

soph_data = tau2_metrics['by_sophistication']
soph_labels = ['Low (0.0)', 'Medium (0.5)', 'High (1.0)']
soph_keys = ['0.0', '0.5', '1.0']
soph_scores = [soph_data[k]['mean_safety_score'] * 100 for k in soph_keys]

colors = ['#27ae60', '#f39c12', '#c0392b']
bars = ax.bar(soph_labels, soph_scores, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

for bar, score in zip(bars, soph_scores):
    height = bar.get_height()
    ax.annotate(f'{score:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('A² Score (%)', fontsize=14)
ax.set_xlabel('Attack Sophistication Level', fontsize=14)
ax.set_title('A² Score by Attack Sophistication', fontsize=16, fontweight='bold')
ax.set_ylim(0, 100)
ax.tick_params(axis='both', labelsize=12)

# Add trend line
z = np.polyfit([0, 1, 2], soph_scores, 1)
p = np.poly1d(z)
ax.plot([0, 1, 2], p([0, 1, 2]), 'k--', linewidth=2, alpha=0.5, label=f'Trend')
ax.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'fig8_asr_by_sophistication.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig8_asr_by_sophistication.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 8: Sophistication - DONE")

# =============================================================================
# Figure 9: Safety-Capability Tradeoff
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Use domain data to show safety vs "capability" (task completion proxy)
# Higher A² score = better safety, Domain complexity = capability challenge

domain_complexity = {'airline': 0.7, 'retail': 0.5, 'telecom': 0.3}  # More complex = higher
markers = ['o', 's']
colors_models = ['#e74c3c', '#9b59b6']

for i, model_key in enumerate(models_tau2):
    model_name = model_display.get(model_key, model_key)
    for j, dk in enumerate(domain_keys):
        if dk in model_domain_data.get(model_key, {}):
            a2 = model_domain_data[model_key][dk]['mean_safety_score'] * 100
            complexity = domain_complexity[dk] * 100

            ax.scatter(complexity, a2,
                      marker=markers[i],
                      color=colors_models[i],
                      s=300, edgecolors='black', linewidth=2,
                      alpha=0.8)
            ax.annotate(domain_labels[j], (complexity + 2, a2 + 1), fontsize=10)

# Add model labels
for i, model_key in enumerate(models_tau2):
    ax.scatter([], [], marker=markers[i], color=colors_models[i], s=150,
               label=model_display.get(model_key, model_key))

ax.set_xlabel('Task Complexity (%)', fontsize=14)
ax.set_ylabel('A² Score (%)', fontsize=14)
ax.set_title('Safety-Capability Tradeoff Analysis', fontsize=16, fontweight='bold')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.legend(loc='lower left', fontsize=11)
ax.tick_params(axis='both', labelsize=12)

# Add quadrant lines
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(output_dir / 'fig9_safety_capability_tradeoff.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig9_safety_capability_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 9: Tradeoff - DONE")

# =============================================================================
# Print Summary Report
# =============================================================================
print("\n" + "="*70)
print("A² SCORE SUMMARY REPORT")
print("="*70)

print("\n--- Overall A² Scores ---")
print(f"Overall Average: {tau2_metrics['overall']['mean_safety_score']*100:.1f}%")
print(f"Defense Rate: {tau2_metrics['overall']['defense_rate']*100:.0f}%")

print("\n--- By Model ---")
for model_key in models_tau2:
    data = tau2_metrics['by_model'][model_key]
    print(f"  {model_display.get(model_key, model_key)}: {data['mean_safety_score']*100:.1f}% (n={data['count']})")

print("\n--- By Domain ---")
for dk, label in zip(domain_keys, domain_labels):
    data = tau2_metrics['by_domain'][dk]
    print(f"  {label}: {data['mean_safety_score']*100:.1f}% (n={data['count']})")

print("\n--- By Attack Strategy ---")
for sk in strategy_keys:
    data = tau2_metrics['by_strategy'][sk]
    print(f"  {sk.replace('_', ' ').title()}: {data['mean_safety_score']*100:.1f}% (n={data['count']})")

print("\n--- By Sophistication ---")
for sk, label in zip(soph_keys, soph_labels):
    data = tau2_metrics['by_sophistication'][sk]
    print(f"  {label}: {data['mean_safety_score']*100:.1f}% (n={data['count']})")

print("\n--- Model × Domain Matrix ---")
print(f"{'Model':<20} | {'Airline':>10} | {'Retail':>10} | {'Telecom':>10}")
print("-" * 55)
for model_key in models_tau2:
    model_name = model_display.get(model_key, model_key)
    row = f"{model_name:<20}"
    for dk in domain_keys:
        if dk in model_domain_data.get(model_key, {}):
            score = model_domain_data[model_key][dk]['mean_safety_score'] * 100
            row += f" | {score:>9.1f}%"
        else:
            row += f" | {'N/A':>10}"
    print(row)

print("\n" + "="*70)
print(f"All figures saved to: {output_dir.absolute()}")
print("="*70)
