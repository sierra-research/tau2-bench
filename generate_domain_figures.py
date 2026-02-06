#!/usr/bin/env python3
"""Generate domain-specific visualizations for the 5 evaluated models."""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

def load_json(path):
    with open(path) as f:
        return json.load(f)

def main():
    results_dir = Path("results/openrouter_eval")
    output_dir = Path("../paper/new_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_json(results_dir / "final_consolidated_all_models.json")

    # Extract all tasks
    all_tasks = []
    for r in results:
        for task in r.get("task_results", []):
            all_tasks.append({
                "model": r["model"],
                "domain": r["domain"],
                "strategy": r["strategy"],
                "sophistication": r["sophistication"],
                "safety_score": task["safety_score"],
                "attack_resisted": task.get("attack_resisted", True),
                "num_violations": task.get("num_violations", 0),
            })

    # Model display names
    model_display = {
        "deepseek-v3.2": "DeepSeek V3.2",
        "gpt-oss-120b": "GPT-OSS-120B",
        "grok-4.1-fast": "Grok 4.1 Fast",
        "kimi-k2.5": "Kimi K2.5",
        "mimo-v2-flash": "Mimo V2 Flash",
    }

    # Domain display names
    domain_display = {
        "airline": "Airline",
        "retail": "Retail",
        "telecom": "Telecom",
    }

    # Strategy display names
    strategy_display = {
        "social_engineering": "Social Eng.",
        "prompt_injection": "Prompt Inj.",
        "policy_exploitation": "Policy Exp.",
        "identity_manipulation": "Identity Man.",
        "information_extraction": "Info Extract.",
    }

    models = sorted(set(t["model"] for t in all_tasks))
    domains = sorted(set(t["domain"] for t in all_tasks))
    strategies = sorted(set(t["strategy"] for t in all_tasks))

    # Calculate scores
    model_scores = defaultdict(list)
    for t in all_tasks:
        model_scores[t["model"]].append(t["safety_score"])
    model_avg = {m: np.mean(scores) for m, scores in model_scores.items()}

    domain_scores = defaultdict(list)
    for t in all_tasks:
        domain_scores[t["domain"]].append(t["safety_score"])

    model_domain_scores = defaultdict(lambda: defaultdict(list))
    for t in all_tasks:
        model_domain_scores[t["model"]][t["domain"]].append(t["safety_score"])

    domain_strategy_scores = defaultdict(lambda: defaultdict(list))
    for t in all_tasks:
        domain_strategy_scores[t["domain"]][t["strategy"]].append(t["safety_score"])

    domain_soph_scores = defaultdict(lambda: defaultdict(list))
    for t in all_tasks:
        domain_soph_scores[t["domain"]][t["sophistication"]].append(t["safety_score"])

    sorted_models = sorted(models, key=lambda m: -model_avg[m])
    sorted_domains = sorted(domains, key=lambda d: -np.mean(domain_scores[d]))

    # Color schemes
    model_colors = {
        "gpt-oss-120b": "#2ecc71",      # Green
        "grok-4.1-fast": "#3498db",     # Blue
        "deepseek-v3.2": "#9b59b6",     # Purple
        "kimi-k2.5": "#e74c3c",         # Red
        "mimo-v2-flash": "#f39c12",     # Orange
    }

    domain_colors = {
        "airline": "#e74c3c",    # Red
        "retail": "#2ecc71",     # Green
        "telecom": "#3498db",    # Blue
    }

    print("=" * 70)
    print("GENERATING DOMAIN-SPECIFIC VISUALIZATIONS")
    print("=" * 70)
    print(f"Total evaluations: {len(all_tasks)}")
    print(f"Models: {len(models)} ({', '.join(model_display.get(m, m) for m in sorted_models)})")
    print(f"Evaluations per model: {len(all_tasks) // len(models)}")

    # =========================================================================
    # Figure 1: Per-Domain Model Comparison (Grouped Bar Chart)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(sorted_domains))
    width = 0.15

    for i, model in enumerate(sorted_models):
        values = [np.mean(model_domain_scores[model][d]) * 100 for d in sorted_domains]
        offset = (i - len(sorted_models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_display.get(model, model),
                     color=model_colors.get(model, f'C{i}'), edgecolor='white', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=8, rotation=0)

    ax.set_xlabel('Domain', fontsize=12)
    ax.set_ylabel('Safety Score (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([domain_display.get(d, d) for d in sorted_domains], fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig9_domain_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "fig9_domain_model_comparison.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig9_domain_model_comparison.png'}")
    plt.close()

    # =========================================================================
    # Figure 2: Domain Vulnerability Breakdown (Stacked by Strategy)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, domain in enumerate(sorted_domains):
        ax = axes[idx]

        # Get model scores for this domain, broken down by strategy
        model_data = []
        for model in sorted_models:
            scores_by_strategy = {}
            for strategy in strategies:
                strat_scores = [t["safety_score"] for t in all_tasks
                               if t["model"] == model and t["domain"] == domain and t["strategy"] == strategy]
                scores_by_strategy[strategy] = np.mean(strat_scores) * 100 if strat_scores else 0
            model_data.append(scores_by_strategy)

        x = np.arange(len(sorted_models))
        width = 0.15

        for j, strategy in enumerate(strategies):
            values = [model_data[i][strategy] for i in range(len(sorted_models))]
            offset = (j - len(strategies)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=strategy_display.get(strategy, strategy))

        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Safety Score (%)', fontsize=10)
        ax.set_title(f'{domain_display.get(domain, domain)} Domain', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([model_display.get(m, m).split()[0] for m in sorted_models],
                          fontsize=8, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        if idx == 2:
            ax.legend(loc='upper right', fontsize=7, ncol=1)

    plt.tight_layout()
    plt.savefig(output_dir / "fig10_domain_strategy_breakdown.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "fig10_domain_strategy_breakdown.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig10_domain_strategy_breakdown.png'}")
    plt.close()

    # =========================================================================
    # Figure 3: Domain-Specific Heatmap (Domain x Strategy x Model)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, domain in enumerate(sorted_domains):
        ax = axes[idx]

        heatmap_data = np.zeros((len(sorted_models), len(strategies)))
        for i, model in enumerate(sorted_models):
            for j, strategy in enumerate(strategies):
                scores = [t["safety_score"] for t in all_tasks
                         if t["model"] == model and t["domain"] == domain and t["strategy"] == strategy]
                heatmap_data[i, j] = np.mean(scores) * 100 if scores else 0

        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels([strategy_display.get(s, s) for s in strategies],
                          rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(sorted_models)))
        ax.set_yticklabels([model_display.get(m, m) for m in sorted_models], fontsize=9)
        ax.set_title(f'{domain_display.get(domain, domain)}', fontsize=12, fontweight='bold')

        # Add text annotations
        for i in range(len(sorted_models)):
            for j in range(len(strategies)):
                val = heatmap_data[i, j]
                color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=axes.ravel().tolist(), label='Safety Score (%)', shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "fig11_domain_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "fig11_domain_heatmaps.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig11_domain_heatmaps.png'}")
    plt.close()

    # =========================================================================
    # Figure 4: Sophistication Impact by Domain
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    soph_levels = [0.0, 0.5, 1.0]
    x = np.arange(len(soph_levels))
    width = 0.25

    for i, domain in enumerate(sorted_domains):
        values = [np.mean(domain_soph_scores[domain][s]) * 100 for s in soph_levels]
        offset = (i - len(sorted_domains)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=domain_display.get(domain, domain),
                     color=domain_colors.get(domain, f'C{i}'), edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Sophistication Level (φ)', fontsize=12)
    ax.set_ylabel('Safety Score (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Low (0.0)', 'Medium (0.5)', 'High (1.0)'], fontsize=11)
    ax.set_ylim(0, 80)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig12_domain_sophistication.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "fig12_domain_sophistication.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig12_domain_sophistication.png'}")
    plt.close()

    # =========================================================================
    # Figure 5: Domain Vulnerability Radar Chart
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(strategies), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for idx, domain in enumerate(sorted_domains):
        ax = axes[idx]

        for model in sorted_models:
            values = []
            for strategy in strategies:
                scores = [t["safety_score"] for t in all_tasks
                         if t["model"] == model and t["domain"] == domain and t["strategy"] == strategy]
                values.append(np.mean(scores) * 100 if scores else 0)
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=1.5, markersize=4,
                   label=model_display.get(model, model), color=model_colors.get(model))
            ax.fill(angles, values, alpha=0.1, color=model_colors.get(model))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([strategy_display.get(s, s) for s in strategies], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(f'{domain_display.get(domain, domain)}', fontsize=12, fontweight='bold', pad=20)

        if idx == 2:
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "fig13_domain_radar.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "fig13_domain_radar.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig13_domain_radar.png'}")
    plt.close()

    # =========================================================================
    # Figure 6: Domain Gap Analysis (Best vs Worst per domain)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    domain_gaps = []
    for domain in sorted_domains:
        model_avgs = {m: np.mean(model_domain_scores[m][domain]) * 100 for m in models}
        best = max(model_avgs.values())
        worst = min(model_avgs.values())
        gap = best - worst
        domain_gaps.append({
            'domain': domain,
            'best': best,
            'worst': worst,
            'gap': gap,
            'best_model': [m for m, v in model_avgs.items() if v == best][0],
            'worst_model': [m for m, v in model_avgs.items() if v == worst][0],
        })

    x = np.arange(len(sorted_domains))
    width = 0.35

    best_vals = [d['best'] for d in domain_gaps]
    worst_vals = [d['worst'] for d in domain_gaps]

    bars1 = ax.bar(x - width/2, best_vals, width, label='Best Model', color='#2ecc71', edgecolor='white')
    bars2 = ax.bar(x + width/2, worst_vals, width, label='Worst Model', color='#e74c3c', edgecolor='white')

    # Add gap annotations
    for i, d in enumerate(domain_gaps):
        ax.annotate('', xy=(i, d['best']), xytext=(i, d['worst']),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(i + 0.15, (d['best'] + d['worst'])/2, f'Δ={d["gap"]:.1f}%',
               fontsize=10, va='center')

    # Add model labels
    for bar, d in zip(bars1, domain_gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               model_display.get(d['best_model'], d['best_model']).split()[0],
               ha='center', va='bottom', fontsize=8, rotation=45)
    for bar, d in zip(bars2, domain_gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               model_display.get(d['worst_model'], d['worst_model']).split()[0],
               ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xlabel('Domain', fontsize=12)
    ax.set_ylabel('Safety Score (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([domain_display.get(d['domain'], d['domain']) for d in domain_gaps], fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig14_domain_gap_analysis.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "fig14_domain_gap_analysis.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig14_domain_gap_analysis.png'}")
    plt.close()

    # =========================================================================
    # Figure 7: Violation Distribution by Domain
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    domain_violations = defaultdict(list)
    for t in all_tasks:
        domain_violations[t["domain"]].append(t.get("num_violations", 0))

    data = [domain_violations[d] for d in sorted_domains]
    bp = ax.boxplot(data, labels=[domain_display.get(d, d) for d in sorted_domains],
                    patch_artist=True)

    for patch, domain in zip(bp['boxes'], sorted_domains):
        patch.set_facecolor(domain_colors.get(domain, 'gray'))
        patch.set_alpha(0.7)

    ax.set_xlabel('Domain', fontsize=12)
    ax.set_ylabel('Number of Violations per Episode', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add mean markers
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(sorted_domains) + 1), means, color='red', marker='D', s=50, zorder=5, label='Mean')
    for i, m in enumerate(means):
        ax.text(i + 1.15, m, f'{m:.1f}', fontsize=9, va='center')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / "fig15_domain_violations.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "fig15_domain_violations.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig15_domain_violations.png'}")
    plt.close()

    print("\n" + "=" * 70)
    print("ALL DOMAIN FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nFigures saved to: {output_dir.absolute()}")
    print("\nGenerated figures:")
    print("  - fig9_domain_model_comparison.png/pdf")
    print("  - fig10_domain_strategy_breakdown.png/pdf")
    print("  - fig11_domain_heatmaps.png/pdf")
    print("  - fig12_domain_sophistication.png/pdf")
    print("  - fig13_domain_radar.png/pdf")
    print("  - fig14_domain_gap_analysis.png/pdf")
    print("  - fig15_domain_violations.png/pdf")

if __name__ == "__main__":
    main()
