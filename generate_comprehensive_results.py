#!/usr/bin/env python3
"""Generate comprehensive results tables and figures for all 5 models."""

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
    figures_dir = Path("../paper/new_figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

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

    models = sorted(set(t["model"] for t in all_tasks))
    domains = sorted(set(t["domain"] for t in all_tasks))
    strategies = sorted(set(t["strategy"] for t in all_tasks))
    sophistications = sorted(set(t["sophistication"] for t in all_tasks))

    # Model display names
    model_display = {
        "deepseek-v3.2": "DeepSeek V3.2",
        "gpt-oss-120b": "GPT-OSS-120B",
        "grok-4.1-fast": "Grok 4.1 Fast",
        "kimi-k2.5": "Kimi K2.5",
        "mimo-v2-flash": "Mimo V2 Flash",
    }

    # Calculate all metrics
    model_scores = defaultdict(list)
    domain_scores = defaultdict(list)
    strategy_scores = defaultdict(list)
    soph_scores = defaultdict(list)
    model_domain_scores = defaultdict(lambda: defaultdict(list))
    model_strategy_scores = defaultdict(lambda: defaultdict(list))
    model_soph_scores = defaultdict(lambda: defaultdict(list))
    domain_strategy_scores = defaultdict(lambda: defaultdict(list))
    strategy_soph_scores = defaultdict(lambda: defaultdict(list))

    for t in all_tasks:
        model_scores[t["model"]].append(t["safety_score"])
        domain_scores[t["domain"]].append(t["safety_score"])
        strategy_scores[t["strategy"]].append(t["safety_score"])
        soph_scores[t["sophistication"]].append(t["safety_score"])
        model_domain_scores[t["model"]][t["domain"]].append(t["safety_score"])
        model_strategy_scores[t["model"]][t["strategy"]].append(t["safety_score"])
        model_soph_scores[t["model"]][t["sophistication"]].append(t["safety_score"])
        domain_strategy_scores[t["domain"]][t["strategy"]].append(t["safety_score"])
        strategy_soph_scores[t["strategy"]][t["sophistication"]].append(t["safety_score"])

    # =========================================================================
    # PRINT TABLES
    # =========================================================================

    print("=" * 80)
    print("τ²-ADV BENCH: COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nTotal evaluations: {len(all_tasks)}")
    print(f"Models: {len(models)}")
    print(f"Domains: {len(domains)}")
    print(f"Attack Strategies: {len(strategies)}")
    print(f"Sophistication Levels: {len(sophistications)}")

    # HIGH-LEVEL: Overall Model Rankings
    print("\n" + "=" * 80)
    print("TABLE 1: OVERALL MODEL SAFETY RANKINGS")
    print("=" * 80)
    print(f"{'Rank':<6} {'Model':<20} {'Safety Score':<15} {'Evaluations':<12}")
    print("-" * 53)
    sorted_models = sorted(models, key=lambda m: -np.mean(model_scores[m]))
    for rank, model in enumerate(sorted_models, 1):
        scores = model_scores[model]
        print(f"{rank:<6} {model_display.get(model, model):<20} {np.mean(scores)*100:>6.1f}%        {len(scores):<12}")

    # HIGH-LEVEL: Domain Rankings
    print("\n" + "=" * 80)
    print("TABLE 2: SAFETY SCORES BY DOMAIN")
    print("=" * 80)
    print(f"{'Domain':<15} {'Safety Score':<15} {'Evaluations':<12}")
    print("-" * 42)
    sorted_domains = sorted(domains, key=lambda d: -np.mean(domain_scores[d]))
    for domain in sorted_domains:
        scores = domain_scores[domain]
        print(f"{domain.capitalize():<15} {np.mean(scores)*100:>6.1f}%        {len(scores):<12}")

    # HIGH-LEVEL: Attack Strategy Rankings
    print("\n" + "=" * 80)
    print("TABLE 3: SAFETY SCORES BY ATTACK STRATEGY")
    print("=" * 80)
    print(f"{'Strategy':<30} {'Safety Score':<15} {'Evaluations':<12}")
    print("-" * 57)
    sorted_strategies = sorted(strategies, key=lambda s: -np.mean(strategy_scores[s]))
    for strategy in sorted_strategies:
        scores = strategy_scores[strategy]
        print(f"{strategy.replace('_', ' ').title():<30} {np.mean(scores)*100:>6.1f}%        {len(scores):<12}")

    # HIGH-LEVEL: Sophistication Level
    print("\n" + "=" * 80)
    print("TABLE 4: SAFETY SCORES BY SOPHISTICATION LEVEL")
    print("=" * 80)
    print(f"{'Level':<15} {'Safety Score':<15} {'Evaluations':<12}")
    print("-" * 42)
    for soph in sorted(sophistications):
        scores = soph_scores[soph]
        level_name = {0.0: "Low (0.0)", 0.5: "Medium (0.5)", 1.0: "High (1.0)"}.get(soph, str(soph))
        print(f"{level_name:<15} {np.mean(scores)*100:>6.1f}%        {len(scores):<12}")

    # LOW-LEVEL: Model × Domain
    print("\n" + "=" * 80)
    print("TABLE 5: MODEL × DOMAIN SAFETY SCORES (%)")
    print("=" * 80)
    header = f"{'Model':<20}" + "".join(f"{d.capitalize():<12}" for d in sorted_domains) + f"{'Average':<12}"
    print(header)
    print("-" * len(header))
    for model in sorted_models:
        row = f"{model_display.get(model, model):<20}"
        for domain in sorted_domains:
            scores = model_domain_scores[model][domain]
            row += f"{np.mean(scores)*100:>6.1f}%     " if scores else f"{'N/A':<12}"
        row += f"{np.mean(model_scores[model])*100:>6.1f}%"
        print(row)

    # LOW-LEVEL: Model × Strategy
    print("\n" + "=" * 80)
    print("TABLE 6: MODEL × ATTACK STRATEGY SAFETY SCORES (%)")
    print("=" * 80)
    strat_abbrev = {
        "social_engineering": "SE",
        "prompt_injection": "PI",
        "policy_exploitation": "PE",
        "identity_manipulation": "IM",
        "information_extraction": "IE",
    }
    header = f"{'Model':<20}" + "".join(f"{strat_abbrev.get(s, s[:3]):<10}" for s in sorted_strategies) + f"{'Avg':<10}"
    print(header)
    print(f"{'Legend: SE=Social Eng, PI=Prompt Inj, PE=Policy Exp, IM=Identity Man, IE=Info Extract'}")
    print("-" * (20 + 10 * len(sorted_strategies) + 10))
    for model in sorted_models:
        row = f"{model_display.get(model, model):<20}"
        for strategy in sorted_strategies:
            scores = model_strategy_scores[model][strategy]
            row += f"{np.mean(scores)*100:>5.1f}%    " if scores else f"{'N/A':<10}"
        row += f"{np.mean(model_scores[model])*100:>5.1f}%"
        print(row)

    # LOW-LEVEL: Model × Sophistication
    print("\n" + "=" * 80)
    print("TABLE 7: MODEL × SOPHISTICATION LEVEL SAFETY SCORES (%)")
    print("=" * 80)
    header = f"{'Model':<20}" + "".join(f"{'φ='+str(s):<12}" for s in sorted(sophistications)) + f"{'Average':<12}"
    print(header)
    print("-" * len(header))
    for model in sorted_models:
        row = f"{model_display.get(model, model):<20}"
        for soph in sorted(sophistications):
            scores = model_soph_scores[model][soph]
            row += f"{np.mean(scores)*100:>6.1f}%     " if scores else f"{'N/A':<12}"
        row += f"{np.mean(model_scores[model])*100:>6.1f}%"
        print(row)

    # LOW-LEVEL: Domain × Strategy
    print("\n" + "=" * 80)
    print("TABLE 8: DOMAIN × ATTACK STRATEGY SAFETY SCORES (%)")
    print("=" * 80)
    header = f"{'Domain':<15}" + "".join(f"{strat_abbrev.get(s, s[:3]):<10}" for s in sorted_strategies) + f"{'Avg':<10}"
    print(header)
    print("-" * (15 + 10 * len(sorted_strategies) + 10))
    for domain in sorted_domains:
        row = f"{domain.capitalize():<15}"
        for strategy in sorted_strategies:
            scores = domain_strategy_scores[domain][strategy]
            row += f"{np.mean(scores)*100:>5.1f}%    " if scores else f"{'N/A':<10}"
        row += f"{np.mean(domain_scores[domain])*100:>5.1f}%"
        print(row)

    # LOW-LEVEL: Strategy × Sophistication
    print("\n" + "=" * 80)
    print("TABLE 9: ATTACK STRATEGY × SOPHISTICATION SAFETY SCORES (%)")
    print("=" * 80)
    header = f"{'Strategy':<30}" + "".join(f"{'φ='+str(s):<12}" for s in sorted(sophistications)) + f"{'Average':<12}"
    print(header)
    print("-" * len(header))
    for strategy in sorted_strategies:
        row = f"{strategy.replace('_', ' ').title():<30}"
        for soph in sorted(sophistications):
            scores = strategy_soph_scores[strategy][soph]
            row += f"{np.mean(scores)*100:>6.1f}%     " if scores else f"{'N/A':<12}"
        row += f"{np.mean(strategy_scores[strategy])*100:>6.1f}%"
        print(row)

    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================

    print("\n" + "=" * 80)
    print("GENERATING FIGURES...")
    print("=" * 80)

    # Figure 1: Overall Model Safety Scores (Horizontal Bar)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    model_names = [model_display.get(m, m) for m in sorted_models]
    model_values = [np.mean(model_scores[m]) * 100 for m in sorted_models]
    colors = plt.cm.RdYlGn(np.array(model_values) / 100)
    bars = ax1.barh(model_names, model_values, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel("Safety Score (%)", fontsize=12)
    ax1.set_xlim(0, 100)
    ax1.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, model_values):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig1_overall_model_scores.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig1_overall_model_scores.pdf", bbox_inches='tight')
    print(f"Saved: {figures_dir / 'fig1_overall_model_scores.png'}")

    # Figure 2: Safety by Domain (Grouped Bar)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(sorted_domains))
    width = 0.15
    for i, model in enumerate(sorted_models):
        values = [np.mean(model_domain_scores[model][d]) * 100 for d in sorted_domains]
        ax2.bar(x + i*width, values, width, label=model_display.get(model, model), alpha=0.8)
    ax2.set_xlabel("Domain", fontsize=12)
    ax2.set_ylabel("Safety Score (%)", fontsize=12)
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels([d.capitalize() for d in sorted_domains])
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig2_domain_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig2_domain_comparison.pdf", bbox_inches='tight')
    print(f"Saved: {figures_dir / 'fig2_domain_comparison.png'}")

    # Figure 3: Safety by Attack Strategy (Grouped Bar)
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    x = np.arange(len(sorted_strategies))
    width = 0.15
    for i, model in enumerate(sorted_models):
        values = [np.mean(model_strategy_scores[model][s]) * 100 for s in sorted_strategies]
        ax3.bar(x + i*width, values, width, label=model_display.get(model, model), alpha=0.8)
    ax3.set_xlabel("Attack Strategy", fontsize=12)
    ax3.set_ylabel("Safety Score (%)", fontsize=12)
    ax3.set_xticks(x + width * 2)
    ax3.set_xticklabels([s.replace('_', '\n').title() for s in sorted_strategies], fontsize=9)
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig3_strategy_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig3_strategy_comparison.pdf", bbox_inches='tight')
    print(f"Saved: {figures_dir / 'fig3_strategy_comparison.png'}")

    # Figure 4: Model × Domain Heatmap
    fig4, ax4 = plt.subplots(figsize=(10, 7))
    heatmap_data = np.zeros((len(sorted_models), len(sorted_domains)))
    for i, model in enumerate(sorted_models):
        for j, domain in enumerate(sorted_domains):
            heatmap_data[i, j] = np.mean(model_domain_scores[model][domain]) * 100
    im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax4.set_xticks(range(len(sorted_domains)))
    ax4.set_xticklabels([d.capitalize() for d in sorted_domains], fontsize=11)
    ax4.set_yticks(range(len(sorted_models)))
    ax4.set_yticklabels([model_display.get(m, m) for m in sorted_models], fontsize=11)
    for i in range(len(sorted_models)):
        for j in range(len(sorted_domains)):
            color = 'white' if heatmap_data[i, j] < 50 else 'black'
            ax4.text(j, i, f"{heatmap_data[i, j]:.1f}", ha='center', va='center', color=color, fontsize=11, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label("Safety Score (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig4_model_domain_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig4_model_domain_heatmap.pdf", bbox_inches='tight')
    print(f"Saved: {figures_dir / 'fig4_model_domain_heatmap.png'}")

    # Figure 5: Model × Strategy Heatmap
    fig5, ax5 = plt.subplots(figsize=(12, 7))
    heatmap_data2 = np.zeros((len(sorted_models), len(sorted_strategies)))
    for i, model in enumerate(sorted_models):
        for j, strategy in enumerate(sorted_strategies):
            heatmap_data2[i, j] = np.mean(model_strategy_scores[model][strategy]) * 100
    im2 = ax5.imshow(heatmap_data2, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax5.set_xticks(range(len(sorted_strategies)))
    ax5.set_xticklabels([s.replace('_', ' ').title() for s in sorted_strategies], rotation=30, ha='right', fontsize=10)
    ax5.set_yticks(range(len(sorted_models)))
    ax5.set_yticklabels([model_display.get(m, m) for m in sorted_models], fontsize=11)
    for i in range(len(sorted_models)):
        for j in range(len(sorted_strategies)):
            color = 'white' if heatmap_data2[i, j] < 50 else 'black'
            ax5.text(j, i, f"{heatmap_data2[i, j]:.1f}", ha='center', va='center', color=color, fontsize=10, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax5)
    cbar2.set_label("Safety Score (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig5_model_strategy_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig5_model_strategy_heatmap.pdf", bbox_inches='tight')
    print(f"Saved: {figures_dir / 'fig5_model_strategy_heatmap.png'}")

    # Figure 6: Sophistication Impact
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sorted(sophistications)))
    width = 0.15
    for i, model in enumerate(sorted_models):
        values = [np.mean(model_soph_scores[model][s]) * 100 for s in sorted(sophistications)]
        ax6.bar(x + i*width, values, width, label=model_display.get(model, model), alpha=0.8)
    ax6.set_xlabel("Sophistication Level (φ)", fontsize=12)
    ax6.set_ylabel("Safety Score (%)", fontsize=12)
    ax6.set_xticks(x + width * 2)
    ax6.set_xticklabels(['Low (0.0)', 'Medium (0.5)', 'High (1.0)'])
    ax6.set_ylim(0, 100)
    ax6.legend(loc='upper right', fontsize=9)
    ax6.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig6_sophistication_impact.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig6_sophistication_impact.pdf", bbox_inches='tight')
    print(f"Saved: {figures_dir / 'fig6_sophistication_impact.png'}")

    # Figure 7: Radar Chart - Model Profiles
    fig7, ax7 = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    categories = [s.replace('_', ' ').title() for s in sorted_strategies]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for model in sorted_models:
        values = [np.mean(model_strategy_scores[model][s]) * 100 for s in sorted_strategies]
        values += values[:1]
        ax7.plot(angles, values, 'o-', linewidth=2, label=model_display.get(model, model))
        ax7.fill(angles, values, alpha=0.1)

    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(categories, fontsize=10)
    ax7.set_ylim(0, 100)
    ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig7_radar_profiles.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig7_radar_profiles.pdf", bbox_inches='tight')
    print(f"Saved: {figures_dir / 'fig7_radar_profiles.png'}")

    # Figure 8: Domain × Strategy Heatmap (aggregated across models)
    fig8, ax8 = plt.subplots(figsize=(12, 6))
    heatmap_data3 = np.zeros((len(sorted_domains), len(sorted_strategies)))
    for i, domain in enumerate(sorted_domains):
        for j, strategy in enumerate(sorted_strategies):
            heatmap_data3[i, j] = np.mean(domain_strategy_scores[domain][strategy]) * 100
    im3 = ax8.imshow(heatmap_data3, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax8.set_xticks(range(len(sorted_strategies)))
    ax8.set_xticklabels([s.replace('_', ' ').title() for s in sorted_strategies], rotation=30, ha='right', fontsize=10)
    ax8.set_yticks(range(len(sorted_domains)))
    ax8.set_yticklabels([d.capitalize() for d in sorted_domains], fontsize=11)
    for i in range(len(sorted_domains)):
        for j in range(len(sorted_strategies)):
            color = 'white' if heatmap_data3[i, j] < 50 else 'black'
            ax8.text(j, i, f"{heatmap_data3[i, j]:.1f}", ha='center', va='center', color=color, fontsize=11, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax8)
    cbar3.set_label("Safety Score (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig8_domain_strategy_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig8_domain_strategy_heatmap.pdf", bbox_inches='tight')
    print(f"Saved: {figures_dir / 'fig8_domain_strategy_heatmap.png'}")

    plt.close('all')
    print("\nAll figures generated successfully!")

if __name__ == "__main__":
    main()
