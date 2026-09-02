# Modal experiment runner

`run_tau_voice.py` runs the existing tau-voice experiment grid on Modal. The
runner is provider-agnostic: providers, models, domains, speech complexities,
user simulators, and user-simulator arguments are all command-line inputs.

Create a private Modal environment, Secrets containing the API keys required
by the selected providers, and (optionally) a Volume. Configure their names
before invoking the launcher:

```bash
export TAU2_MODAL_APP_NAME=tau2-experiments
export TAU2_MODAL_VOLUME_NAME=tau2-experiment-results
export TAU2_MODAL_SECRET_NAMES=tau2-experiment-keys,tau2-experiment-review-key

modal run --detach --env tau-bench \
  src/experiments/modal/run_tau_voice.py::run \
  --providers openai:gpt-realtime \
  --domains retail,airline,telecom \
  --complexities regular \
  --result-root /results/my-run
```

The launcher has no web endpoint. Results stay in the configured private Modal
Volume. API credentials are supplied through two Modal Secrets (evaluation and
review) and are excluded from the uploaded repository image.

To intentionally continue an existing checkpoint, pass `--resume`. Without
that flag, the launcher refuses to reuse a non-empty result directory.

Pine models use the normal OpenAI provider syntax:

```bash
--providers openai:pine-voice-preview
```

Set `PINE_API_KEY` and `PINE_REALTIME_BASE_URL` in the Modal Secret. The OpenAI
realtime provider selects them automatically for model names beginning with
`pine-`.
