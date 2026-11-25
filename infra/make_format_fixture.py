"""M0 / D8 — capture the EXACT format tau2 sends the retail agent.

Train, sample, and eval must all see byte-identical inputs. This dumps the
retail `llm_agent` system prompt and the OpenAI-format tool schemas (built the
same way `tau2 run` builds them), plus the non-thinking decode config we pin for
Qwen3-8B, into `infra/format_fixture.json`. Downstream code asserts against this
file so format drift (the #1 silent failure) is caught immediately.

Run (no GPU): `uv run python infra/make_format_fixture.py`
Verify later runs match: `uv run python infra/make_format_fixture.py --check`
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tau2.agent.llm_agent import LLMAgent
from tau2.domains.retail.environment import get_environment

FIXTURE_PATH = Path(__file__).with_name("format_fixture.json")
FORMAT_VERSION = "retail-qwen3-8b-nonthinking-v1"

# Non-thinking decode config for Qwen3-8B, passed through litellm -> vLLM.
# enable_thinking=False is the documented Qwen3 chat-template switch; pinning it
# here keeps sampling and eval in the same mode (Simia: thinking -> ~0 on tau2).
QWEN3_NONTHINKING_LLM_ARGS = {
    "temperature": 0.0,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}


def build_fixture() -> dict:
    env = get_environment()
    tools = env.get_tools()
    # Built exactly as tau2 constructs the agent for a retail run.
    agent = LLMAgent(tools=tools, domain_policy=env.policy, llm="placeholder")

    tool_schemas = [t.openai_schema for t in tools]
    system_prompt = agent.system_prompt

    return {
        "format_version": FORMAT_VERSION,
        "domain": "retail",
        "agent_implementation": "llm_agent",
        "thinking_mode": "non_thinking",
        "llm_args_agent": QWEN3_NONTHINKING_LLM_ARGS,
        "system_prompt": system_prompt,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "tool_names": sorted(t["function"]["name"] for t in tool_schemas),
        "num_tools": len(tool_schemas),
        "tool_schemas": tool_schemas,
        "tool_schemas_sha256": hashlib.sha256(
            json.dumps(tool_schemas, sort_keys=True).encode()
        ).hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Fail if the current format differs from the committed fixture.")
    args = ap.parse_args()

    current = build_fixture()

    if args.check:
        if not FIXTURE_PATH.exists():
            raise SystemExit(f"No fixture at {FIXTURE_PATH}; run without --check first.")
        saved = json.loads(FIXTURE_PATH.read_text())
        drift = [
            key for key in ("system_prompt_sha256", "tool_schemas_sha256", "llm_args_agent")
            if saved.get(key) != current.get(key)
        ]
        if drift:
            raise SystemExit(f"FORMAT DRIFT in {drift} — sampling/training/eval would diverge.")
        print(f"OK: format matches {FIXTURE_PATH.name} ({current['format_version']})")
        return

    FIXTURE_PATH.write_text(json.dumps(current, indent=2))
    print(f"Wrote {FIXTURE_PATH}")
    print(f"  format_version : {current['format_version']}")
    print(f"  tools          : {current['num_tools']} ({', '.join(current['tool_names'])})")
    print(f"  system_prompt  : {len(current['system_prompt'])} chars "
          f"(sha {current['system_prompt_sha256'][:12]})")
    print(f"  tool_schemas   : sha {current['tool_schemas_sha256'][:12]}")
    print(f"  llm_args_agent : {json.dumps(current['llm_args_agent'])}")


if __name__ == "__main__":
    main()
