"""Run a custom LLM judge over tau2 simulations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

import litellm
from tau2.utils.utils import DATA_DIR

_POLICY_CACHE: dict[str, str] = {}
POLICY_PATHS: dict[str, Path] = {
    "telecom": DATA_DIR / "tau2" / "domains" / "telecom" / "main_policy.md",
    "airline": DATA_DIR / "tau2" / "domains" / "airline" / "policy.md",
    "retail": DATA_DIR / "tau2" / "domains" / "retail" / "policy.md",
}
POLICY_PATHS: dict[str, Path] = {
    "telecom": DATA_DIR / "tau2" / "domains" / "telecom" / "main_policy.md",
    "airline": DATA_DIR / "tau2" / "domains" / "airline" / "policy.md",
    "retail": DATA_DIR / "tau2" / "domains" / "retail" / "policy.md",
}

PROMPT_PRESETS: dict[str, dict[str, str]] = {
    "policy_conversation_v1": {
        "description": (
            "Judge sees domain policy plus full transcript (current default)."
        ),
        "system_prompt": (
            "You are a meticulous {domain_name} quality inspector. "
            "Given a task definition and a transcript, decide if the agent met the "
            "policy requirements."
        ),
        "user_template": (
            "Domain: {domain_name}\n\n"
            "Domain policy:\n{policy_text}\n\n"
            "Conversation transcript:\n{conversation}\n\n"
            "Judging instructions:\n"
            "- Base your judgment only on the domain policy above and the conversation "
            "transcript (including tool call outputs).\n"
            "- Decide whether the agent resolved the user's issue while following policy. "
            "If crucial steps are missing or policy was violated, return `fail` and explain why.\n"
            "- Keep reasoning evidence-based: cite specific turns or tool results when describing issues.\n"
            "Return a JSON object with keys `verdict` (\"pass\" or \"fail\"), `justification` "
            "(a concise explanation tied to evidence), and `issues` "
            "(an array of concrete problems or missing steps).\n"
        ),
    },
    "policy_chain_of_thought_v1": {
        "description": (
            "Detailed review: emphasizes step-by-step evidence from the transcript and tool calls."
        ),
        "system_prompt": (
            "You are a senior QA analyst for the {domain_name} domain. "
            "You must only use the provided transcript and policy."
        ),
        "user_template": (
            "Domain: {domain_name}\n\n"
            "Domain policy:\n{policy_text}\n\n"
            "Conversation transcript:\n{conversation}\n\n"
            "Judging instructions:\n"
            "- Reconstruct the agent's behavior step-by-step, citing turns or tool calls when justifying decisions.\n"
            "- Confirm whether the agent satisfied the user's request and respected every policy rule.\n"
            "- Highlight any missed troubleshooting steps, policy violations, or premature transfers.\n"
            "- Return JSON with `verdict`, `justification`, `issues` (same schema as the default prompt). "
            "Ensure the justification mentions the specific steps that support the verdict.\n"
        ),
    },
    "customer_outcome_v1": {
        "description": (
            "Focuses on user satisfaction and closure quality, referencing the final user turns."
        ),
        "system_prompt": (
            "You are a {domain_name} customer-experience reviewer. "
            "Assess whether the user left satisfied and if the agent's closing matched policy."
        ),
        "user_template": (
            "Domain: {domain_name}\n\n"
            "Domain policy (for reference):\n{policy_text}\n\n"
            "Conversation transcript:\n{conversation}\n\n"
            "Judging instructions:\n"
            "- Pay special attention to the final user messages and whether the user's problem was clearly solved.\n"
            "- Verify the agent provided accurate next steps or transfer messaging when required.\n"
            "- Fail the conversation if the agent ended without confirmation, gave misleading info, or ignored policy.\n"
            "- Return JSON with `verdict`, `justification`, and `issues`.\n"
        ),
    },
    "quick_triage_v1": {
        "description": (
            "Shallow triage prompt for fast disagreement detection (minimal reasoning)."
        ),
        "system_prompt": (
            "You are a {domain_name} triage reviewer. "
            "Give a quick pass/fail based on obvious success or failure."
        ),
        "user_template": (
            "Domain: {domain_name}\n\n"
            "Conversation transcript:\n{conversation}\n\n"
            "Judging instructions:\n"
            "- Do a lightweight check: Did the agent clearly fix the user's issue while respecting policy tone? "
            "If anything critical is missing or obviously wrong, mark `fail`.\n"
            "- Otherwise mark `pass`. Keep justification short (one sentence) and list the biggest issue if failing.\n"
            "- Return JSON with `verdict`, `justification`, and `issues`.\n"
        ),
    },
}

def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text())

def _load_domain_policy(domain_name: str) -> str:
    if domain_name in _POLICY_CACHE:
        return _POLICY_CACHE[domain_name]
    base_dir = DATA_DIR / "tau2" / "domains" / domain_name
    if domain_name in POLICY_PATHS:
        policy_path = POLICY_PATHS[domain_name]
    else:
        policy_path = base_dir / "main_policy.md"
    if policy_path.exists():
        policy_text = policy_path.read_text()
    else:
        raise FileNotFoundError(
            f"Policy file not found for domain '{domain_name}' at {policy_path}."
        )
    _POLICY_CACHE[domain_name] = policy_text
    return policy_text


def _format_message(msg: dict[str, Any]) -> Iterable[str]:
    role = msg.get("role")
    content = msg.get("content")
    if role == "assistant":
        if content:
            yield f"Assistant: {content}"
        tool_calls = msg.get("tool_calls") or []
        for call in tool_calls:
            fn = call.get("name")
            args = call.get("arguments")
            yield f"Assistant TOOL CALL -> {fn} {args}"
    elif role == "user":
        if content:
            yield f"User: {content}"
    elif role == "tool":
        yield f"Tool {msg.get('id', '')}: {content}"
    elif role == "system" and content:
        yield f"System: {content}"


def _format_conversation(simulation: dict[str, Any]) -> str:
    lines: list[str] = []
    for msg in simulation.get("messages", []):
        lines.extend(_format_message(msg))
    if not lines:
        return "(no dialogue)"
    return "\n".join(f"{idx+1}. {line}" for idx, line in enumerate(lines))


def _render_prompt(
    preset_name: str, *, domain_name: str, policy_text: str, conversation: str
) -> tuple[str, str]:
    if preset_name not in PROMPT_PRESETS:
        raise ValueError(
            f"Unknown judge prompt preset '{preset_name}'. "
            f"Available: {list(PROMPT_PRESETS.keys())}"
        )
    preset = PROMPT_PRESETS[preset_name]
    system_prompt = preset["system_prompt"].format(domain_name=domain_name)
    policy_section = policy_text.strip() if policy_text else "(policy unavailable)"
    user_prompt = preset["user_template"].format(
        domain_name=domain_name,
        policy_text=policy_section,
        conversation=conversation,
    )
    return system_prompt, user_prompt


def _call_judge(model: str, system_prompt: str, prompt: str, temperature: float) -> str:
    response = litellm.completion(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} judge record(s) to {output_path}")


def _is_tau2_success(simulation: dict[str, Any]) -> bool:
    reward = simulation.get("reward_info", {}).get("reward")
    if reward is None:
        return False
    return abs(reward - 1.0) <= 1e-6


def _parse_judge_response(response: str) -> dict[str, Any]:
    """
    Parse the judge response, which should be a JSON object.
    Returns a dictionary with keys: verdict, justification, issues, raw.
    """
    parsed: dict[str, Any] = {}
    candidate = response.strip()
    if candidate:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if 0 <= start < end:
                try:
                    parsed = json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    parsed = {}
    verdict = str(parsed.get("verdict", "unknown")).lower()
    justification = parsed.get("justification")
    issues = parsed.get("issues")
    if not isinstance(issues, list):
        issues = issues if isinstance(issues, str) else ""
        issues = [issues] if issues else []
    return {
        "verdict": verdict,
        "justification": justification,
        "issues": issues,
        "raw": response,
    }


def _normalize_verdict_bool(verdict: str | None) -> tuple[str | None, bool | None]:
    if verdict is None:
        return None, None
    cleaned = verdict.strip().lower()
    if not cleaned:
        return verdict, None
    positive = {"pass", "satisfied", "success", "resolved"}
    negative = {"fail", "unsatisfied", "error", "rejected"}
    if cleaned in positive:
        return verdict, True
    if cleaned in negative:
        return verdict, False
    return verdict, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a custom GPT-4o judge on τ² simulations."
    )
    parser.add_argument(
        "--simulation-files",
        nargs="+",
        required=True,
        help="One or more simulation result JSON files (e.g., outputs of `tau2 run`).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels matching the simulation files (defaults to file stems).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain name (airline, retail, telecom, mock, etc.).",
    )
    parser.add_argument(
        "--simulation-count",
        type=int,
        default=None,
        help="Randomly sample this many simulations (after filtering) before judging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used when sampling simulations.",
    )
    parser.add_argument(
        "--model",
        default="azure/gpt-4o-2024-11-20",
        help="Judge model name passed to LiteLLM (default: azure/gpt-4o-2024-11-20).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the judge model.",
    )
    parser.add_argument(
        "--judge-prompt",
        choices=sorted(PROMPT_PRESETS.keys()),
        help="(Deprecated) Single judge prompt preset to use.",
    )
    parser.add_argument(
        "--judge-prompts",
        choices=sorted(PROMPT_PRESETS.keys()),
        nargs="+",
        help="One or more judge prompt presets to run. "
        "If multiple are provided, each prompt reuses the same sampled simulations.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to store the judge outputs (JSONL).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of simulations to judge.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Where to store the summary results JSON (default: <output>.summary.json).",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    summary_output = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )
    simulation_paths = [
        Path(path).expanduser().resolve() for path in args.simulation_files
    ]
    if args.labels is not None and len(args.labels) != len(simulation_paths):
        raise ValueError("Number of labels must match number of simulation files.")
    labels = args.labels or [path.stem for path in simulation_paths]
    cli_domain = args.domain
    prompt_names: list[str] = []
    if args.judge_prompts:
        prompt_names = args.judge_prompts
    elif args.judge_prompt:
        prompt_names = [args.judge_prompt]
    else:
        raise ValueError("You must specify --judge-prompts (or the legacy --judge-prompt).")

    for prompt_name in prompt_names:
        description = PROMPT_PRESETS[prompt_name]["description"]
        print(f"Using judge prompt '{prompt_name}': {description}")

    judge_records: list[dict[str, Any]] = []
    summary_entries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for sim_path, label in zip(simulation_paths, labels):
        try:
            data = _load_json(sim_path)
        except FileNotFoundError as exc:
            print(f"Skipping {sim_path}: {exc}")
            continue

        file_domain = (
            data.get("info", {})
            .get("environment_info", {})
            .get("domain_name")
        )
        if file_domain is not None and cli_domain != file_domain:
            print(
                f"Skipping {sim_path}: domain mismatch "
                f"(expected {cli_domain}, found {file_domain})."
            )
            continue
        domain_name = cli_domain
        tasks = data.get("tasks") or []
        tasks_by_id = {task.get("id"): task for task in tasks if task.get("id")}
        policy_text = _load_domain_policy(domain_name)

        for simulation in data.get("simulations", []):
            task_id = simulation.get("task_id")
            task = tasks_by_id.get(task_id)
            if task is None:
                print(
                    f"Skipping simulation {simulation.get('id')} in {sim_path} "
                    f"because task_id {task_id} was not found."
                )
                continue

            candidates.append(
                {
                    "file": str(sim_path),
                    "label": label,
                    "task": task,
                    "simulation": simulation,
                    "domain": domain_name,
                    "policy": policy_text,
                }
            )

    if not candidates:
        print("No simulations found to judge.")
        return

    rng = random.Random(args.seed)
    if args.simulation_count is not None:
        count = min(args.simulation_count, len(candidates))
        if count < args.simulation_count:
            print(
                f"Requested {args.simulation_count} simulations but only "
                f"{len(candidates)} available. Judging {count} instead."
            )
        candidates = rng.sample(candidates, count)

    if args.limit is not None:
        candidates = candidates[: args.limit]

    selected_candidates = candidates

    def make_output_paths(prompt_name: str) -> tuple[Path, Path]:
        if len(prompt_names) == 1:
            return output_path, summary_output
        suffix = f"_{prompt_name}"
        raw = output_path.with_name(output_path.stem + suffix + output_path.suffix)
        summ = summary_output.with_name(
            summary_output.stem + suffix + summary_output.suffix
        )
        return raw, summ

    for prompt_name in prompt_names:
        prompt_description = PROMPT_PRESETS[prompt_name]["description"]
        print(f"\n=== Running prompt '{prompt_name}' ({prompt_description}) ===")
        prompt_records: list[dict[str, Any]] = []
        prompt_summary_entries: list[dict[str, Any]] = []
        raw_path, summary_path = make_output_paths(prompt_name)
        total_prompt_cost = 0.0

        for candidate in selected_candidates:
            conversation = _format_conversation(candidate["simulation"])
            system_prompt, prompt = _render_prompt(
                prompt_name,
                domain_name=candidate["domain"],
                policy_text=candidate["policy"],
                conversation=conversation,
            )
            response = _call_judge(
                model=args.model,
                system_prompt=system_prompt,
                prompt=prompt,
                temperature=args.temperature,
            )
            judge_response = response.choices[0].message["content"]
            usage = getattr(response, "usage", None)
            cost = None
            if usage is not None:
                cost = litellm.completion_cost(response)
                if cost is not None:
                    total_prompt_cost += cost

            record = {
                "file": candidate["file"],
                "label": candidate["label"],
                "task_id": candidate["task"].get("id"),
                "simulation_id": candidate["simulation"].get("id"),
                "tau2_reward": candidate["simulation"].get("reward_info", {}).get("reward"),
                "judge_response": judge_response,
                "prompt": prompt,
                "judge_prompt": prompt,
                "judge_system_prompt": system_prompt,
                "judge_model": args.model,
                "judge_temperature": args.temperature,
                "judge_prompt_name": prompt_name,
                "judge_prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "judge_completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "judge_total_tokens": getattr(usage, "total_tokens", None) if usage else None,
                "judge_cost_usd": cost,
            }
            prompt_records.append(record)

            tau2_success = _is_tau2_success(candidate["simulation"])
            parsed = _parse_judge_response(judge_response)
            _, judge_pass = _normalize_verdict_bool(parsed["verdict"])
            agreement = (
                None
                if judge_pass is None
                else (tau2_success == judge_pass)
            )
            prompt_summary_entries.append(
                {
                    "file": candidate["file"],
                    "label": candidate["label"],
                    "simulation_id": candidate["simulation"].get("id"),
                    "task_id": candidate["task"].get("id"),
                    "tau2_reward": candidate["simulation"].get("reward_info", {}).get(
                        "reward"
                    ),
                    "tau2_success": tau2_success,
                    "judge_verdict": parsed["verdict"],
                    "judge_justification": parsed["justification"],
                    "judge_issues": parsed["issues"],
                    "agreement": agreement,
                    "judge_prompt": prompt,
                    "judge_system_prompt": system_prompt,
                    "judge_model": args.model,
                    "judge_temperature": args.temperature,
                    "judge_prompt_name": prompt_name,
                    "judge_prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "judge_completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                    "judge_total_tokens": getattr(usage, "total_tokens", None) if usage else None,
                    "judge_cost_usd": cost,
                }
            )

        _write_jsonl(prompt_records, raw_path)
        summary = {
            "total_judged": len(prompt_summary_entries),
            "agree_count": sum(1 for entry in prompt_summary_entries if entry["agreement"]),
            "disagree_count": sum(
                1 for entry in prompt_summary_entries if not entry["agreement"]
            ),
            "entries": prompt_summary_entries,
            "total_cost_usd": total_prompt_cost,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary to {summary_path}")
        print(
            f"Prompt '{prompt_name}' cost: ${total_prompt_cost:.4f} "
            f"({len(prompt_summary_entries)} simulations)"
        )


if __name__ == "__main__":
    main()

