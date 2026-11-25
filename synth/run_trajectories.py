"""Generate trajectories over the synthetic tasks using tau2's own runner.

This is a thin wrapper: tau2 already runs the agent vs the user simulator and
computes+stores `reward_info` (DB + COMMUNICATE) in a tau2-compliant Results
file. We only register the task set and hand tau2 a config. Legality filtering
(D11) is a separate post-step — see filter_legal.py.

Run later (needs an agent endpoint + API key):
  uv run python synth/run_trajectories.py \
      --agent-llm hosted_vllm/qwen3-8b --api-base http://localhost:8000/v1 \
      --num-trials 4 --max-concurrency 8 --save-to retail_synth_qwen
"""

from __future__ import annotations

import argparse

from synth_tasks import register
from tau2.data_model.simulation import TextRunConfig
from tau2.run import run_domain


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent-llm", required=True,
                    help="Agent model (litellm id), e.g. hosted_vllm/qwen3-8b or claude-...")
    ap.add_argument("--api-base", default=None, help="OpenAI-compatible base URL for the agent.")
    ap.add_argument("--user-llm", default=None, help="User-simulator model (defaults to tau2 default).")
    ap.add_argument("--num-trials", type=int, default=4, help="k rollouts per task (temp>0 for diversity).")
    ap.add_argument("--temperature", type=float, default=1.0, help="Agent sampling temperature.")
    ap.add_argument("--num-tasks", type=int, default=None, help="Limit number of tasks (smoke tests).")
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--save-to", default="retail_synth", help="Run name under data/simulations/.")
    args = ap.parse_args()

    task_set = register()

    # Non-thinking + endpoint wiring travel as litellm kwargs (see infra/format_fixture.json).
    agent_args = {"temperature": args.temperature,
                  "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if args.api_base:
        agent_args["api_base"] = args.api_base

    cfg = dict(
        domain="retail",
        task_set_name=task_set,
        task_split_name=None,
        agent="llm_agent",
        llm_agent=args.agent_llm,
        llm_args_agent=agent_args,
        num_trials=args.num_trials,
        num_tasks=args.num_tasks,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        save_to=args.save_to,
    )
    if args.user_llm:
        cfg["llm_user"] = args.user_llm

    config = TextRunConfig(**cfg)
    results = run_domain(config)
    n = len(results.simulations)
    print(f"\nGenerated {n} simulations over task set '{task_set}'. "
          f"Filter with: uv run python synth/filter_legal.py "
          f"data/simulations/{args.save_to}/results.json")


if __name__ == "__main__":
    main()
