"""Cost-per-resolution reporting for Return-and-Exchange τ-bench runs."""

from __future__ import annotations

from tau2.data_model.message import AssistantMessage, Message, UserMessage
from tau2.data_model.simulation import Results, SimulationRun
from tau2.metrics.agent_metrics import is_successful


def _message_tokens(message: Message) -> tuple[int, int]:
    usage = message.usage or {}
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def usage_from_messages(
    messages: list[Message], *, user_cost: float | None = None
) -> dict:
    """Aggregate token/cost usage from a simulation trajectory."""
    agent_in = agent_out = 0
    agent_cost = 0.0
    supervisor_in = supervisor_out = 0
    supervisor_cost = 0.0

    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        usage = message.usage or {}
        agent_in += int(usage.get("prompt_tokens", 0))
        agent_out += int(usage.get("completion_tokens", 0))
        supervisor_in += int(usage.get("supervisor_input_tokens", 0))
        supervisor_out += int(usage.get("supervisor_output_tokens", 0))
        supervisor_cost += float(usage.get("supervisor_cost_usd", 0.0))
        if message.cost is not None:
            agent_cost += float(message.cost) - float(
                usage.get("supervisor_cost_usd", 0.0)
            )

    user_sim_cost = float(user_cost or 0.0)
    total_cost = agent_cost + supervisor_cost + user_sim_cost
    by_component = {
        "agent": {
            "calls": sum(
                1
                for m in messages
                if isinstance(m, AssistantMessage)
                and (m.usage or {}).get("prompt_tokens")
            ),
            "input_tokens": agent_in,
            "output_tokens": agent_out,
            "cost_usd": agent_cost,
        },
        "supervisor": {
            "calls": sum(
                1
                for m in messages
                if isinstance(m, AssistantMessage)
                and (m.usage or {}).get("supervisor_input_tokens")
            ),
            "input_tokens": supervisor_in,
            "output_tokens": supervisor_out,
            "cost_usd": supervisor_cost,
        },
    }
    if user_sim_cost:
        user_in = user_out = 0
        for message in messages:
            if isinstance(message, UserMessage) and not message.is_tool_call():
                prompt, completion = _message_tokens(message)
                user_in += prompt
                user_out += completion
        by_component["user_simulator"] = {
            "calls": sum(
                1
                for m in messages
                if isinstance(m, UserMessage)
                and not m.is_tool_call()
                and m.cost is not None
            ),
            "input_tokens": user_in,
            "output_tokens": user_out,
            "cost_usd": user_sim_cost,
        }

    return {
        "api_calls": by_component["agent"]["calls"]
        + by_component["supervisor"]["calls"]
        + by_component.get("user_simulator", {}).get("calls", 0),
        "input_tokens": agent_in
        + supervisor_in
        + by_component.get("user_simulator", {}).get("input_tokens", 0),
        "output_tokens": agent_out
        + supervisor_out
        + by_component.get("user_simulator", {}).get("output_tokens", 0),
        "cost_usd": total_cost,
        "by_component": by_component,
    }


def usage_for_simulation(sim: SimulationRun) -> dict:
    messages = sim.messages or []
    usage = usage_from_messages(messages, user_cost=sim.user_cost)
    usage["resolved"] = bool(
        sim.reward_info is not None and is_successful(sim.reward_info.reward)
    )
    usage["task_id"] = sim.task_id
    usage["trial"] = sim.trial
    return usage


def aggregate_usage(sims: list[SimulationRun]) -> dict:
    details = [usage_for_simulation(sim) for sim in sims]
    if not details:
        return {
            "runs": 0,
            "passed_runs": 0,
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "passed_cost_usd": 0.0,
            "cost_per_resolution_usd": 0.0,
            "cost_per_pass_usd": None,
            "by_component": {},
        }

    passed = [d for d in details if d["resolved"]]
    total_cost = sum(d["cost_usd"] for d in details)
    passed_cost = sum(d["cost_usd"] for d in passed)
    by_component: dict[str, dict] = {}
    for detail in details:
        for comp, stats in detail["by_component"].items():
            bucket = by_component.setdefault(
                comp,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            bucket["calls"] += stats["calls"]
            bucket["input_tokens"] += stats["input_tokens"]
            bucket["output_tokens"] += stats["output_tokens"]
            bucket["cost_usd"] += stats["cost_usd"]

    return {
        "runs": len(details),
        "passed_runs": len(passed),
        "api_calls": sum(d["api_calls"] for d in details),
        "input_tokens": sum(d["input_tokens"] for d in details),
        "output_tokens": sum(d["output_tokens"] for d in details),
        "cost_usd": total_cost,
        "passed_cost_usd": passed_cost,
        "cost_per_resolution_usd": total_cost / len(details),
        "cost_per_pass_usd": passed_cost / len(passed) if passed else None,
        "by_component": by_component,
        "per_simulation": details,
    }


def annotate_results(results: Results) -> dict:
    """Attach per-simulation token usage to sim.info and return aggregate summary."""
    summary = aggregate_usage(results.simulations)
    per_sim = summary.pop("per_simulation")
    for sim, usage in zip(results.simulations, per_sim, strict=False):
        sim.info = sim.info or {}
        sim.info["token_usage"] = usage
    return summary


def print_cost_summary(usage: dict, *, num_trials: int | None = None) -> None:
    k_label = f"k={num_trials}" if num_trials else "all trials"
    print("\n" + "=" * 50)
    print("COST PER RESOLUTION")
    print("=" * 50)
    print(
        f"  Total: ${usage['cost_usd']:.4f}  "
        f"({usage['input_tokens']:,} in / {usage['output_tokens']:,} out, "
        f"{usage['api_calls']} billed LLM calls across {usage['runs']} conversations)"
    )
    print(
        f"  Per conversation (avg over {k_label}): "
        f"${usage['cost_per_resolution_usd']:.4f}"
    )
    if usage["cost_per_pass_usd"] is not None:
        print(
            f"  Per resolved case (reward=1): ${usage['cost_per_pass_usd']:.4f} "
            f"({usage['passed_runs']}/{usage['runs']} resolved)"
        )
    else:
        print("  Per resolved case: n/a (no passes)")
    for comp in ("agent", "supervisor", "user_simulator"):
        stats = usage["by_component"].get(comp)
        if not stats or stats["cost_usd"] <= 0:
            continue
        print(
            f"  {comp:14s} ${stats['cost_usd']:.4f}  "
            f"({stats['calls']} calls, {stats['input_tokens']:,} in / "
            f"{stats['output_tokens']:,} out)"
        )
