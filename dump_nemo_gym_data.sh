uv venv --python 3.12 .venv --allow-existing
source .venv/bin/activate
uv sync --active

tau2 run \
    --domain airline \
    --agent-llm dummy \
    --agent-llm-args "{\"api_base\": \"dummy\", \"api_key\": \"EMPTY\"}" \
    --user-llm dummy \
    --num-trials 1 \
    --max-retries 1 \
    --max-concurrency 256 \
    --seed 42 \
    --save-to $(pwd)/results/airline \
    --auto-resume

tau2 run \
    --domain telecom \
    --agent-llm dummy \
    --agent-llm-args "{\"api_base\": \"dummy\", \"api_key\": \"EMPTY\"}" \
    --user-llm dummy \
    --num-trials 1 \
    --max-retries 1 \
    --max-concurrency 256 \
    --seed 42 \
    --save-to $(pwd)/results/telecom \
    --auto-resume

tau2 run \
    --domain retail \
    --agent-llm dummy \
    --agent-llm-args "{\"api_base\": \"dummy\", \"api_key\": \"EMPTY\"}" \
    --user-llm dummy \
    --num-trials 1 \
    --max-retries 1 \
    --max-concurrency 256 \
    --seed 42 \
    --save-to $(pwd)/results/retail \
    --auto-resume
