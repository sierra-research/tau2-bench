#!/usr/bin/env python3
"""
Generate per-task reward summary table for different domains.

Usage:
    python src/tau2/scripts/per_task_summary.py --domain retail
    python src/tau2/scripts/per_task_summary.py --domain telecom
    python src/tau2/scripts/per_task_summary.py --domain airline

Generating annotation sets (retail, 2026-03-10):

  # Voice Fragile: text passed but control failed (control results only)
  uv run python src/experiments/tau_voice/annotation/export_html.py \
    --batch-name 2026-03-10_voice_fragile \
    --results tmp/organized_results_mar10/voice/voice_trial1/base/retail_control_gemini_gemini-live-2.5-flash-native-audio/results.json \
             tmp/organized_results_mar10/voice/voice_trial1/base/retail_control_openai_gpt-realtime-1.5/results.json \
             tmp/organized_results_mar10/voice/voice_trial1/base/retail_control_xai_xai-realtime/results.json \
    --filter-tasks 6,7,8,14,19,22,23,24,25,28,31,33,35,36,51,56,59,79,87,106 \
    --filter-reward "< 1"

  # Noise Fragile: control passed but regular failed (regular results only)
  uv run python src/experiments/tau_voice/annotation/export_html.py \
    --batch-name 2026-03-10_noise_fragile \
    --results tmp/organized_results_mar10/voice/voice_trial1/base/retail_regular_gemini_gemini-live-2.5-flash-native-audio/results.json \
             tmp/organized_results_mar10/voice/voice_trial1/base/retail_regular_openai_gpt-realtime-1.5/results.json \
             tmp/organized_results_mar10/voice/voice_trial1/base/retail_regular_xai_xai-realtime/results.json \
    --filter-tasks 0,16,29,32,42,46,48,58,66,76,80,81,83,89,94,98,101,108,113 \
    --filter-reward "< 1"
"""

import argparse
from pathlib import Path

from tau2.utils.io_utils import load_results_dict


def get_rewards(data: dict) -> dict:
    """Extract task_id -> reward mapping from results data."""
    rewards = {}
    for s in data.get("simulations", []):
        task_id = s.get("task_id")
        ri = s.get("reward_info")
        reward = ri.get("reward", 0) if ri else 0
        rewards[str(task_id)] = reward
    return rewards


def load_results(path: str) -> dict:
    """Load results from path (supports both JSON and directory formats)."""
    return load_results_dict(path)


def get_domain_config(domain: str) -> tuple[dict, str]:
    """Get experiment paths and output filename for a domain."""
    if domain == "retail":
        text = Path("tmp/organized_results_mar09/text")
        voice = Path("tmp/organized_results_mar10/voice/voice_trial1/base")
        experiments = {
            "GPT-4.1 (text)": text / "retail_text_gpt41/results.json",
            "GPT-5.2-high (text)": text / "retail_text_gpt52-high/results.json",
            "Gem ctrl": voice
            / "retail_control_gemini_gemini-live-2.5-flash-native-audio/results.json",
            "OAI ctrl": voice / "retail_control_openai_gpt-realtime-1.5/results.json",
            "XAI ctrl": voice / "retail_control_xai_xai-realtime/results.json",
            "Gem reg": voice
            / "retail_regular_gemini_gemini-live-2.5-flash-native-audio/results.json",
            "OAI reg": voice / "retail_regular_openai_gpt-realtime-1.5/results.json",
            "XAI reg": voice / "retail_regular_xai_xai-realtime/results.json",
        }
        output_file = "retail_per_task_summary.txt"
    elif domain == "telecom":
        text = Path("tmp/organized_results_mar09/text")
        voice = Path("tmp/organized_results_mar10/voice/voice_trial1/base")
        experiments = {
            "GPT-4.1 (text)": text / "telecom_text_gpt41/results.json",
            "GPT-5.2 (text)": text / "telecom_text_gpt52/results.json",
            "Gem ctrl": voice
            / "telecom_control_gemini_gemini-live-2.5-flash-native-audio/results.json",
            "OAI ctrl": voice / "telecom_control_openai_gpt-realtime-1.5/results.json",
            "XAI ctrl": voice / "telecom_control_xai_xai-realtime/results.json",
            "Gem reg": voice
            / "telecom_regular_gemini_gemini-live-2.5-flash-native-audio/results.json",
            "OAI reg": voice / "telecom_regular_openai_gpt-realtime-1.5/results.json",
            "XAI reg": voice / "telecom_regular_xai_xai-realtime/results.json",
        }
        output_file = "telecom_per_task_summary.txt"
    elif domain == "airline":
        text = Path("tmp/organized_results_mar09/text")
        voice = Path("tmp/organized_results_mar10/voice/voice_trial1/base")
        experiments = {
            "GPT-4.1 (text)": text / "airline_text_gpt41/results.json",
            "GPT-5.2-high (text)": text / "airline_text_gpt52-high/results.json",
            "Gem ctrl": voice
            / "airline_control_gemini_gemini-live-2.5-flash-native-audio/results.json",
            "OAI ctrl": voice / "airline_control_openai_gpt-realtime-1.5/results.json",
            "XAI ctrl": voice / "airline_control_xai_xai-realtime/results.json",
            "Gem reg": voice
            / "airline_regular_gemini_gemini-live-2.5-flash-native-audio/results.json",
            "OAI reg": voice / "airline_regular_openai_gpt-realtime-1.5/results.json",
            "XAI reg": voice / "airline_regular_xai_xai-realtime/results.json",
        }
        output_file = "airline_per_task_summary.txt"
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return experiments, output_file


def generate_summary(domain: str):
    experiments, output_file = get_domain_config(domain)

    rewards = {}
    for name, path in experiments.items():
        if path.exists():
            rewards[name] = get_rewards(load_results(path))
        else:
            print(f"Warning: {path} not found")
            rewards[name] = {}

    # Get all task IDs - preserve order from first results file (reflects run order)
    all_tasks = []
    seen = set()
    # Use the first text experiment as the reference for ordering
    first_exp_path = list(experiments.values())[0]
    if first_exp_path.exists():
        first_data = load_results_dict(first_exp_path)
        for s in first_data.get("simulations", []):
            tid = str(s.get("task_id"))
            if tid not in seen:
                all_tasks.append(tid)
                seen.add(tid)

    # Add any tasks from other experiments that weren't in the first
    for r in rewards.values():
        for tid in r.keys():
            if tid not in seen:
                all_tasks.append(tid)
                seen.add(tid)

    # Column widths
    col_names = list(experiments.keys())
    col_widths = {name: max(len(name), 5) for name in col_names}

    # Identify column types
    text_cols = [n for n in col_names if "text" in n.lower()]
    ctrl_cols = [n for n in col_names if "ctrl" in n.lower()]
    reg_cols = [n for n in col_names if "reg" in n.lower()]
    audio_cols = ctrl_cols + reg_cols  # All non-text are audio
    if not audio_cols:
        audio_cols = [n for n in col_names if "audio" in n.lower()]

    # Create table
    lines = []
    lines.append(f"# Per-Task Reward Summary - {domain.capitalize()} Domain")
    lines.append("")

    # Determine task ID width based on domain
    max_task_len = max(len(str(t)) for t in all_tasks) if all_tasks else 7
    task_width = min(max_task_len, 80)  # Cap at 80 chars

    # Header
    header = f"| {'Task ID':<{task_width}} |"
    separator = "|" + "-" * (task_width + 2) + "|"
    for name in col_names:
        w = col_widths[name]
        header += f" {name:^{w}} |"
        separator += "-" * (w + 2) + "|"
    header += " Label                    |"
    separator += "--------------------------|"
    # Add second label column for ctrl vs reg comparison if both exist
    has_ctrl_reg = len(ctrl_cols) > 0 and len(reg_cols) > 0
    if has_ctrl_reg:
        header += " Ctrl vs Reg             |"
        separator += "-------------------------|"
    lines.append(header)
    lines.append(separator)

    # Counters for labels (separate for label1 and label2)
    label1_counts = {}
    label2_counts = {}
    task_labels1 = {}
    task_labels2 = {}

    # Data rows
    for task_id in all_tasks:
        # Truncate long task IDs for display
        display_id = str(task_id)[:task_width]
        row = f"| {display_id:<{task_width}} |"
        for name in col_names:
            r = rewards[name].get(task_id, "-")
            w = col_widths[name]
            val = f"{r:.1f}" if isinstance(r, float) else str(r)
            row += f" {val:^{w}} |"

        # Calculate label (text vs control audio only)
        text_passed = sum(1 for n in text_cols if rewards[n].get(task_id, 0) == 1.0)
        # For first label, only compare against control audio (not regular)
        ctrl_audio_cols = ctrl_cols if ctrl_cols else audio_cols
        ctrl_audio_passed = sum(
            1 for n in ctrl_audio_cols if rewards[n].get(task_id, 0) == 1.0
        )

        # Check for XAI-only pass (XAI ctrl passes, all others fail)
        xai_ctrl_cols = [n for n in ctrl_audio_cols if "XAI" in n or "xai" in n.lower()]
        xai_ctrl_passed = sum(
            1 for n in xai_ctrl_cols if rewards[n].get(task_id, 0) == 1.0
        )
        non_xai_ctrl = [n for n in ctrl_audio_cols if n not in xai_ctrl_cols]
        non_xai_ctrl_passed = sum(
            1 for n in non_xai_ctrl if rewards[n].get(task_id, 0) == 1.0
        )

        label = ""
        if text_passed == len(text_cols):  # Both text passed
            if ctrl_audio_passed == 0:
                label = "TEXT_ONLY"
            elif ctrl_audio_passed == 1:
                label = "TEXT+1_CTRL"
            elif ctrl_audio_passed == len(ctrl_audio_cols):
                label = "ALL_PASS"
            else:
                label = f"TEXT+{ctrl_audio_passed}_CTRL"
        elif text_passed == 0 and ctrl_audio_passed == 0:
            label = "ALL_FAIL"
        elif text_passed == 0 and xai_ctrl_passed > 0 and non_xai_ctrl_passed == 0:
            label = "XAI_ONLY"
        else:
            label = ""

        label1_counts[label] = label1_counts.get(label, 0) + 1
        task_labels1[task_id] = label
        row += f" {label:<24} |"

        # Second label: ctrl vs reg comparison
        if has_ctrl_reg:
            ctrl_passed = sum(1 for n in ctrl_cols if rewards[n].get(task_id, 0) == 1.0)
            reg_passed = sum(1 for n in reg_cols if rewards[n].get(task_id, 0) == 1.0)
            # Only label if reg has data for this task
            reg_has_data = any(rewards[n].get(task_id) is not None for n in reg_cols)

            label2 = ""
            if reg_has_data:
                diff = ctrl_passed - reg_passed
                if diff >= 3:
                    label2 = "CTRL+3"
                elif diff == 2:
                    label2 = "CTRL+2"
                elif diff <= -3:
                    label2 = "REG+3"
                elif diff == -2:
                    label2 = "REG+2"

            label2_counts[label2] = label2_counts.get(label2, 0) + 1
            task_labels2[task_id] = label2
            row += f" {label2:<23} |"

        lines.append(row)

    # Summary
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model                 | Pass Rate       |")
    lines.append("|-----------------------|-----------------|")
    for name in col_names:
        r = rewards[name]
        if r:
            passed = sum(1 for v in r.values() if v == 1.0)
            total = len(r)
            pct = passed / total * 100 if total else 0
            lines.append(f"| {name:<21} | {passed:>3}/{total:<3} ({pct:>5.1f}%) |")

    # Label 1 distribution (text vs control audio)
    lines.append("")
    lines.append("## Label Distribution (Text vs Control)")
    lines.append("")
    lines.append("| Label                    | Count |")
    lines.append("|--------------------------|-------|")
    for label in [
        "TEXT_ONLY",
        "TEXT+1_CTRL",
        "TEXT+2_CTRL",
        "TEXT+3_CTRL",
        "ALL_PASS",
        "ALL_FAIL",
        "XAI_ONLY",
        "",
    ]:
        if label in label1_counts:
            display = label if label else "(other)"
            lines.append(f"| {display:<24} | {label1_counts[label]:>5} |")

    # Label 2 distribution (ctrl vs reg) - only if applicable
    if has_ctrl_reg:
        lines.append("")
        lines.append("## Label Distribution (Ctrl vs Reg)")
        lines.append("")
        lines.append("| Label                    | Count |")
        lines.append("|--------------------------|-------|")
        for label in ["CTRL+3", "CTRL+2", "REG+2", "REG+3", ""]:
            if label in label2_counts:
                display = label if label else "(no diff >= 2)"
                lines.append(f"| {display:<24} | {label2_counts[label]:>5} |")

    output = "\n".join(lines)

    # Write to file
    output_path = Path("tmp/organized_results_mar10") / output_file
    with open(output_path, "w") as f:
        f.write(output)

    print(f"Saved to {output_path}")
    print()
    print("Summary:")
    for name in col_names:
        r = rewards[name]
        if r:
            passed = sum(1 for v in r.values() if v == 1.0)
            total = len(r)
            pct = passed / total * 100 if total else 0
            print(f"  {name}: {passed}/{total} ({pct:.1f}%)")

    voice_fragile = sorted(
        [
            t
            for t, label in task_labels1.items()
            if label in ("TEXT_ONLY", "TEXT+1_CTRL")
        ],
        key=int,
    )
    noise_fragile = sorted(
        [t for t, label in task_labels2.items() if label in ("CTRL+2", "CTRL+3")],
        key=int,
    )

    print()
    print(f"Voice Fragile ({len(voice_fragile)} tasks):")
    print(",".join(voice_fragile))
    print()
    print(f"Noise Fragile ({len(noise_fragile)} tasks):")
    print(",".join(noise_fragile))


def main():
    parser = argparse.ArgumentParser(description="Generate per-task reward summary")
    parser.add_argument(
        "--domain",
        type=str,
        default="retail",
        choices=["retail", "telecom", "airline"],
        help="Domain to generate summary for",
    )
    args = parser.parse_args()

    generate_summary(args.domain)


if __name__ == "__main__":
    main()
