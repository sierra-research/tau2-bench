#!/usr/bin/env python3
"""Generate final visualizations for all 5 models."""

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
            })

    # Calculate metrics
    models = sorted(set(t["model"] for t in all_tasks))
    domains = sorted(set(t["domain"] for t in all_tasks))
    strategies = sorted(set(t["strategy"] for t in all_tasks))

    # Model names for display
    model_display = {
        "deepseek-v3.2": "DeepSeek V3.2",
        "gpt-oss-120b": "GPT-OSS-120B",
        "grok-4.1-fast": "Grok 4.1 Fast",
        "kimi-k2.5": "Kimi K2.5",
        "mimo-v2-flash": "Mimo V2 Flash",
    }

    # Calculate safety scores by model
    model_scores = defaultdict(list)
    for t in all_tasks:
        model_scores[t["model"]].append(t["safety_score"])

    model_avg = {m: np.mean(scores) for m, scores in model_scores.items()}

    # Calculate scores by domain
    domain_scores = defaultdict(list)
    for t in all_tasks:
        domain_scores[t["domain"]].append(t["safety_score"])

    # Calculate scores by strategy
    strategy_scores = defaultdict(list)
    for t in all_tasks:
        strategy_scores[t["strategy"]].append(t["safety_score"])

    # Calculate model x domain scores
    model_domain_scores = defaultdict(lambda: defaultdict(list))
    for t in all_tasks:
        model_domain_scores[t["model"]][t["domain"]].append(t["safety_score"])

    # Calculate model x strategy scores
    model_strategy_scores = defaultdict(lambda: defaultdict(list))
    for t in all_tasks:
        model_strategy_scores[t["model"]][t["strategy"]].append(t["safety_score"])

    # Print summary
    print("=" * 70)
    print("τ²-ADV BENCH: FINAL EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nTotal task results: {len(all_tasks)}")
    print(f"Models evaluated: {len(models)}")

    print("\n" + "=" * 70)
    print("SAFETY SCORES BY MODEL")
    print("=" * 70)
    for model in sorted(models, key=lambda m: -model_avg[m]):
        scores = model_scores[model]
        print(f"  {model_display.get(model, model):20s}: {np.mean(scores)*100:5.1f}% (n={len(scores)})")

    print("\n" + "=" * 70)
    print("SAFETY SCORES BY DOMAIN")
    print("=" * 70)
    for domain in sorted(domains, key=lambda d: -np.mean(domain_scores[d])):
        scores = domain_scores[domain]
        print(f"  {domain.capitalize():20s}: {np.mean(scores)*100:5.1f}%")

    print("\n" + "=" * 70)
    print("SAFETY SCORES BY STRATEGY")
    print("=" * 70)
    for strategy in sorted(strategies, key=lambda s: -np.mean(strategy_scores[s])):
        scores = strategy_scores[strategy]
        print(f"  {strategy.replace('_', ' ').title():25s}: {np.mean(scores)*100:5.1f}%")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Overall safety scores by model (bar chart)
    ax1 = axes[0, 0]
    model_names = [model_display.get(m, m) for m in sorted(models, key=lambda m: -model_avg[m])]
    model_values = [model_avg[m] * 100 for m in sorted(models, key=lambda m: -model_avg[m])]
    colors = plt.cm.RdYlGn(np.array(model_values) / 100)
    bars = ax1.barh(model_names, model_values, color=colors)
    ax1.set_xlabel("Safety Score (%)")
    ax1.set_title("Overall Safety Score by Model")
    ax1.set_xlim(0, 100)
    for bar, val in zip(bars, model_values):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va='center')

    # 2. Safety scores by domain (bar chart)
    ax2 = axes[0, 1]
    domain_names = [d.capitalize() for d in sorted(domains, key=lambda d: -np.mean(domain_scores[d]))]
    domain_values = [np.mean(domain_scores[d]) * 100 for d in sorted(domains, key=lambda d: -np.mean(domain_scores[d]))]
    colors = plt.cm.RdYlGn(np.array(domain_values) / 100)
    bars = ax2.barh(domain_names, domain_values, color=colors)
    ax2.set_xlabel("Safety Score (%)")
    ax2.set_title("Safety Score by Domain")
    ax2.set_xlim(0, 100)
    for bar, val in zip(bars, domain_values):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va='center')

    # 3. Model x Domain heatmap
    ax3 = axes[1, 0]
    heatmap_data = np.zeros((len(models), len(domains)))
    sorted_models = sorted(models, key=lambda m: -model_avg[m])
    sorted_domains = sorted(domains, key=lambda d: -np.mean(domain_scores[d]))
    for i, model in enumerate(sorted_models):
        for j, domain in enumerate(sorted_domains):
            if model_domain_scores[model][domain]:
                heatmap_data[i, j] = np.mean(model_domain_scores[model][domain]) * 100

    im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax3.set_xticks(range(len(sorted_domains)))
    ax3.set_xticklabels([d.capitalize() for d in sorted_domains])
    ax3.set_yticks(range(len(sorted_models)))
    ax3.set_yticklabels([model_display.get(m, m) for m in sorted_models])
    ax3.set_title("Model × Domain Safety Scores")
    for i in range(len(sorted_models)):
        for j in range(len(sorted_domains)):
            ax3.text(j, i, f"{heatmap_data[i, j]:.0f}%", ha='center', va='center',
                    color='white' if heatmap_data[i, j] < 50 else 'black')
    plt.colorbar(im, ax=ax3, label="Safety Score (%)")

    # 4. Safety scores by strategy (bar chart)
    ax4 = axes[1, 1]
    strategy_names = [s.replace('_', ' ').title() for s in sorted(strategies, key=lambda s: -np.mean(strategy_scores[s]))]
    strategy_values = [np.mean(strategy_scores[s]) * 100 for s in sorted(strategies, key=lambda s: -np.mean(strategy_scores[s]))]
    colors = plt.cm.RdYlGn(np.array(strategy_values) / 100)
    bars = ax4.barh(strategy_names, strategy_values, color=colors)
    ax4.set_xlabel("Safety Score (%)")
    ax4.set_title("Safety Score by Attack Strategy")
    ax4.set_xlim(0, 100)
    for bar, val in zip(bars, strategy_values):
        ax4.text(val + 1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va='center')

    plt.tight_layout()
    plt.savefig(results_dir / "final_5_models_results.png", dpi=150, bbox_inches='tight')
    plt.savefig(results_dir / "final_5_models_results.pdf", bbox_inches='tight')
    print(f"\nSaved: {results_dir / 'final_5_models_results.png'}")

    # Create Model x Strategy heatmap
    fig2, ax = plt.subplots(figsize=(12, 6))
    sorted_strategies = sorted(strategies, key=lambda s: -np.mean(strategy_scores[s]))
    heatmap_data2 = np.zeros((len(sorted_models), len(sorted_strategies)))
    for i, model in enumerate(sorted_models):
        for j, strategy in enumerate(sorted_strategies):
            if model_strategy_scores[model][strategy]:
                heatmap_data2[i, j] = np.mean(model_strategy_scores[model][strategy]) * 100

    im2 = ax.imshow(heatmap_data2, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(sorted_strategies)))
    ax.set_xticklabels([s.replace('_', ' ').title() for s in sorted_strategies], rotation=45, ha='right')
    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels([model_display.get(m, m) for m in sorted_models])
    ax.set_title("Model × Strategy Safety Scores")
    for i in range(len(sorted_models)):
        for j in range(len(sorted_strategies)):
            ax.text(j, i, f"{heatmap_data2[i, j]:.0f}%", ha='center', va='center',
                    color='white' if heatmap_data2[i, j] < 50 else 'black')
    plt.colorbar(im2, ax=ax, label="Safety Score (%)")
    plt.tight_layout()
    plt.savefig(results_dir / "final_model_strategy_heatmap.png", dpi=150, bbox_inches='tight')
    plt.savefig(results_dir / "final_model_strategy_heatmap.pdf", bbox_inches='tight')
    print(f"Saved: {results_dir / 'final_model_strategy_heatmap.png'}")

    plt.close('all')

if __name__ == "__main__":
    main()
