# tau-Elicitation reviewer evidence

This directory is the compact, reviewer-facing evidence archive for the tau-Elicitation paper. It contains all 5,970 scored transcript records, all 6,422 available utterance-level speech-judge outputs, final human labels, exact run configurations and prompt objects, deterministic example calls, and the checked analysis inputs. The large audio/tick corpus remains a detached evidence root and is linked by SHA-256.

## Contents

- `results/manifest.csv`: one row per frozen results root.
- `run_configs/`: exact recorded configuration for every cell.
- `prompts/`: content-addressed agent policies, caller guidelines, exact historical task snapshots, and the v6 speech-judge prompt.
- `transcripts/`: one compact JSONL file per results root, including agent tools and silent caller-side spelling/read-back events.
- `examples/`: deterministic calls spanning both strategies and three systems.
- `speech_judgments/`: all 6,422 exported utterance judgments: 4,948 from the paper's main speech cohort and 1,474 supplemental judgments, including retained/excluded findings and errors.
- `judge_validation/`: direct utterance/call mapping between judge output and final human labels.
- `analysis_inputs/`: crossed outcomes, rollups, deterministic complication draws, caller-effort ledger, deterministic behavioral recomputation, the 210-row matched one-field composition ledger, caller-voice and main significance results, and speech rollup.
- `audit.json`: executable claim checks.

`SOURCE_GAPS.md` records any remaining source or statistical-analysis re-execution gaps; both workflows in the release comparison have frozen sources.

The prompt manifest distinguishes the caller guideline actually selected by the frozen simulation builders from a stale inbound guideline stored in the historical top-level run metadata. The content-addressed runtime prompt is the one used for reproduction.

## Verify

```bash
tau2 paper elicitation-verify --root papers/tau-intake/v1/reproduction
# With the detached 41 GB source corpus:
tau2 paper elicitation-verify --root papers/tau-intake/v1/reproduction --evidence-root /path/to/tau-elicit
```

Verification is offline and makes no model calls. To exercise the runnable benchmark on one frozen task instead, set the required provider keys and run:

```bash
tau2 run --domain intake_free --audio-native --audio-native-provider openai --audio-native-model gpt-realtime-2 --reasoning-effort xhigh --user-llm gpt-5.5 --user-llm-args '{"reasoning_effort":"xhigh","temperature":0.0}' --complication-rate 1.0 --channel-effects-mode regular --speech-effects-mode regular --seed 9401 --task-ids intake_medications_hard_04 --num-trials 1 --max-concurrency 1 --timeout 1200 --save-to tau_elicitation_smoke
```

To regenerate the v6 speech judgments from the detached audio corpus, write into a new directory (frozen paper roots are protected):

```bash
tau2 judges rejudge /path/to/tau-elicit/main_runs/ONE_CELL --delivery --delivery-only --rejudge --delivery-sample-rate 1.0 --delivery-model gemini/gemini-3.1-pro-preview --max-concurrency 8 --output /path/to/rejudged/ONE_CELL
# Add --limit-sims 1 --max-segments 1 for a one-API-call smoke test.
```

The paired significance analysis is also self-contained in the compact archive:

```bash
uv run --extra experiments python src/experiments/intake/main_significance.py --repo-root .
uv run --extra experiments python src/experiments/intake/realism_effects.py --repo-root .
uv run --extra experiments python src/experiments/intake/caller_voice_significance.py --repo-root .
python src/experiments/intake/release_claims.py --check
```

Current claim-audit status: **PASS**. See `AUDIT.md`.
