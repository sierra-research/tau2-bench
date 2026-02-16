uv run tau2 run \
    --domain airline \
    --agent llm_agent_pctx \
    --agent-llm "openrouter/openai/gpt-5" \
    --user-llm "openrouter/openai/gpt-4o-2024-05-13" \
    --log-level INFO \
    --task-ids 2
    # 0 1 2 3 4 5 6 7 8 9 10