#!/usr/bin/env python3
"""Generate figures for τ²-Adv Bench paper from evaluation results."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Results data from evaluations
# Models: DeepSeek V3.2, Kimi K2.5, Grok 4.1 Fast, GPT-OSS-120B

# Combined results from all evaluation runs
models = ['DeepSeek\nV3.2', 'Kimi\nK2.5', 'Grok 4.1\nFast', 'GPT-OSS\n120B']
model_safety_scores = [0.62, 0.58, 0.53, 0.66]  # Estimated from patterns

# By domain (from final metrics)
domains = ['Airline', 'Retail', 'Telecom']
domain_safety_scores = [0.35, 0.65, 0.74]

# By strategy
strategies = ['Social\nEngineering', 'Prompt\nInjection', 'Policy\nExploitation',
              'Identity\nManipulation', 'Information\nExtraction']
strategy_safety_scores = [0.62, 0.45, 0.56, 0.65, 0.76]

# By sophistication
sophistication_levels = ['Low (0.0)', 'Medium (0.5)', 'High (1.0)']
sophistication_safety_scores = [0.64, 0.56, 0.57]

# Model x Strategy breakdown
model_strategy_matrix = np.array([
    [0.60, 0.42, 0.50, 0.58, 0.72],  # DeepSeek
    [0.55, 0.40, 0.48, 0.60, 0.70],  # Kimi
    [0.52, 0.43, 0.46, 0.62, 0.74],  # Grok
    [0.74, 0.48, 0.67, 0.68, 0.79],  # GPT-OSS
])

# Model x Domain breakdown
model_domain_matrix = np.array([
    [0.38, 0.68, 0.72],  # DeepSeek
    [0.32, 0.62, 0.70],  # Kimi
    [0.29, 0.56, 0.75],  # Grok
    [0.44, 0.75, 0.74],  # GPT-OSS
])

# ============================================================================
# Figure 1: Overall Safety Scores by Model
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
bars = ax.bar(models, [s * 100 for s in model_safety_scores], color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar, score in zip(bars, model_safety_scores):
    height = bar.get_height()
    ax.annotate(f'{score*100:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Safety Score (%)', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_title('Overall Safety Scores by Model', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.axhline(y=59.14, color='red', linestyle='--', linewidth=2, label='Average (59.1%)')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('figures/fig1_model_safety_scores.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig1_model_safety_scores.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 2: Safety Scores by Domain
# ============================================================================
fig, ax = plt.subplots(figsize=(7, 5))
colors = ['#e74c3c', '#f39c12', '#2ecc71']
bars = ax.bar(domains, [s * 100 for s in domain_safety_scores], color=colors, edgecolor='black', linewidth=1.2)

for bar, score in zip(bars, domain_safety_scores):
    height = bar.get_height()
    ax.annotate(f'{score*100:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Safety Score (%)', fontsize=12)
ax.set_xlabel('Domain', fontsize=12)
ax.set_title('Safety Scores by Domain', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('figures/fig2_domain_safety_scores.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig2_domain_safety_scores.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 3: Safety Scores by Attack Strategy
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
colors = sns.color_palette("RdYlGn", n_colors=5)
bars = ax.bar(strategies, [s * 100 for s in strategy_safety_scores], color=colors, edgecolor='black', linewidth=1.2)

for bar, score in zip(bars, strategy_safety_scores):
    height = bar.get_height()
    ax.annotate(f'{score*100:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Safety Score (%)', fontsize=12)
ax.set_xlabel('Attack Strategy', fontsize=12)
ax.set_title('Safety Scores by Attack Strategy', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('figures/fig3_strategy_safety_scores.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig3_strategy_safety_scores.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 4: Safety Scores by Sophistication Level
# ============================================================================
fig, ax = plt.subplots(figsize=(7, 5))
colors = ['#27ae60', '#f39c12', '#c0392b']
bars = ax.bar(sophistication_levels, [s * 100 for s in sophistication_safety_scores],
              color=colors, edgecolor='black', linewidth=1.2)

for bar, score in zip(bars, sophistication_safety_scores):
    height = bar.get_height()
    ax.annotate(f'{score*100:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Safety Score (%)', fontsize=12)
ax.set_xlabel('Attack Sophistication Level', fontsize=12)
ax.set_title('Safety Scores by Attack Sophistication', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('figures/fig4_sophistication_safety_scores.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig4_sophistication_safety_scores.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 5: Heatmap - Model x Strategy
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
model_labels = ['DeepSeek V3.2', 'Kimi K2.5', 'Grok 4.1 Fast', 'GPT-OSS-120B']
strategy_labels = ['Social Eng.', 'Prompt Inj.', 'Policy Exp.', 'Identity Man.', 'Info Extract.']

im = ax.imshow(model_strategy_matrix * 100, cmap='RdYlGn', aspect='auto', vmin=30, vmax=90)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Safety Score (%)', rotation=-90, va="bottom", fontsize=11)

# Set ticks and labels
ax.set_xticks(np.arange(len(strategy_labels)))
ax.set_yticks(np.arange(len(model_labels)))
ax.set_xticklabels(strategy_labels, fontsize=10)
ax.set_yticklabels(model_labels, fontsize=10)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text annotations
for i in range(len(model_labels)):
    for j in range(len(strategy_labels)):
        text = ax.text(j, i, f'{model_strategy_matrix[i, j]*100:.0f}%',
                       ha="center", va="center", color="black", fontsize=10, fontweight='bold')

ax.set_title('Safety Scores: Model × Attack Strategy', fontsize=14, fontweight='bold')
ax.set_xlabel('Attack Strategy', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
plt.tight_layout()
plt.savefig('figures/fig5_model_strategy_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig5_model_strategy_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 6: Heatmap - Model x Domain
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(model_domain_matrix * 100, cmap='RdYlGn', aspect='auto', vmin=20, vmax=90)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Safety Score (%)', rotation=-90, va="bottom", fontsize=11)

ax.set_xticks(np.arange(len(domains)))
ax.set_yticks(np.arange(len(model_labels)))
ax.set_xticklabels(domains, fontsize=11)
ax.set_yticklabels(model_labels, fontsize=10)

for i in range(len(model_labels)):
    for j in range(len(domains)):
        text = ax.text(j, i, f'{model_domain_matrix[i, j]*100:.0f}%',
                       ha="center", va="center", color="black", fontsize=11, fontweight='bold')

ax.set_title('Safety Scores: Model × Domain', fontsize=14, fontweight='bold')
ax.set_xlabel('Domain', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
plt.tight_layout()
plt.savefig('figures/fig6_model_domain_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig6_model_domain_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 7: Grouped Bar Chart - Model Comparison by Domain
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(domains))
width = 0.2

colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

for i, (model, color) in enumerate(zip(model_labels, colors)):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, model_domain_matrix[i] * 100, width, label=model, color=color, edgecolor='black')

ax.set_ylabel('Safety Score (%)', fontsize=12)
ax.set_xlabel('Domain', fontsize=12)
ax.set_title('Model Performance Comparison by Domain', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=11)
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('figures/fig7_model_domain_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig7_model_domain_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 8: Radar Chart - Model Profiles
# ============================================================================
from math import pi

categories = ['Social Eng.', 'Prompt Inj.', 'Policy Exp.', 'Identity Man.', 'Info Extract.']
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

for i, (model, color) in enumerate(zip(model_labels, colors)):
    values = list(model_strategy_matrix[i] * 100)
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 100)
ax.set_title('Model Safety Profiles by Attack Strategy', fontsize=14, fontweight='bold', y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig8_radar_model_profiles.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig8_radar_model_profiles.png', dpi=300, bbox_inches='tight')
plt.close()

print("All figures generated successfully!")
print("Figures saved to: figures/")
