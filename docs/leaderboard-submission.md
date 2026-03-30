# Leaderboard Submission Guide

Submit your agent results to the τ-bench leaderboard at **[taubench.com](https://taubench.com)**.

| Modality | Description | Guide |
|----------|-------------|-------|
| **Text** | Standard text-based half-duplex evaluation | This page (below) |
| **Voice** | Audio-native full-duplex evaluation (τ-voice) | [Voice Submissions](#voice-submissions-τ-voice) |

> **Submitting voice results?** Voice evaluation requires specialized infrastructure and is handled case-by-case. Skip to the [Voice Submissions](#voice-submissions-τ-voice) section.

---

## Text Submissions

### Requirements

Your submission should meet these constraints:

1. **Domain coverage** — we recommend including results for all three core domains (`retail`, `airline`, `telecom`). You may submit results for a single domain, but the leaderboard's "Overall" column only appears when all four domains (including `banking_knowledge`) have Pass^1 scores
2. **Consistent model configuration** — all trajectory files must use the same agent LLM and user simulator LLM with identical arguments across all domains
3. **One result per domain** — each domain should appear exactly once
4. **All tasks completed** — run evaluation on all tasks within each domain (don't use `--task-ids` or `--num-tasks` filters)
5. **4+ trials** — we strongly prefer results with at least 4 trials per domain for statistical reliability

> **Note**: Use the `base` task split (default) when evaluating your agent to ensure you're testing on the complete, standard task set consistent with the original τ-bench methodology.

## Submission Types: Standard vs Custom

The leaderboard distinguishes between two types of submissions:

### Standard Submissions (Default)

Standard submissions evaluate a **general-purpose LLM** using the **default τ-bench scaffold**:
- A general-purpose LLM as the agent (not specifically trained for this benchmark)
- The standard tool set provided by τ-bench
- Default prompts and evaluation protocol
- No modifications to the evaluation setup

If you're evaluating an off-the-shelf LLM using τ-bench as documented without modifications, your submission is **standard**. You don't need to specify `submission_type` in your JSON (it defaults to `"standard"`).

### Custom Submissions

Custom submissions include **any approach that differs from the standard evaluation**, such as:

**Modified Scaffolds:**
- Multi-model routers or model ensembles
- Additional tools beyond the standard τ-bench tool set
- Modified agent orchestration or control flow
- Modified prompts or system instructions

**Domain-Specific Training:**
- Models trained or fine-tuned specifically on τ-bench domains (airline, retail, telecom customer service)
- Models trained using τ-bench tasks, reward signals, or evaluation data
- Models where training data significantly overlaps with τ-bench evaluation scenarios

Custom submissions **must** include detailed methodology documentation:

1. **Set `submission_type` to `"custom"`** in your `submission.json`
2. **Provide comprehensive `methodology.notes`** explaining what modifications were made, why, and how the custom system works at a high level
3. **Link to your implementation** in the `references` array (GitHub repo, paper, blog post)
4. **Set `methodology.verification.modified_prompts` to `true`** if you modified any prompts

## Step 1: Run Evaluations

Run your agent on all domains with consistent settings:

```bash
tau2 run --domain retail --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_retail
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_airline
tau2 run --domain telecom --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_telecom
```

**Important**: Use identical `--agent-llm`, `--user-llm`, and their arguments across all runs. You can use any LLM as the user simulator, but this choice will be reported on the leaderboard. We recommend using `gpt-5.2` as the user simulator for the most accurate results.

Trajectory files are saved in `data/simulations/`.

### Banking Knowledge Domain

The banking domain requires a retrieval configuration for the knowledge base. You must include a `retrieval_config` field in your `banking_knowledge` results specifying which retrieval method was used.

```bash
# Banking domain with terminal-based retrieval
tau2 run --domain banking_knowledge --retrieval-config terminal_use --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4

# Banking domain with BM25 retrieval
tau2 run --domain banking_knowledge --retrieval-config bm25 --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4
```

Common `retrieval_config` values:

| Config | Description |
|--------|-------------|
| `terminal_use` | Agent navigates the knowledge base via shell commands (grep, cat, find, etc.) |
| `openai_embeddings` | Dense retrieval using OpenAI's text-embedding-3-large model |
| `qwen_embeddings` | Dense retrieval using Qwen3-Embedding model |
| `bm25` | Sparse retrieval using BM25 |

If you use a different retrieval method, use a short descriptive string (e.g., `"custom_reranker"`) and document it in `methodology.notes`. The `retrieval_config` value is displayed as a badge on the leaderboard.

### Cost Tracking (Optional but Recommended)

To enable fair comparisons between models with different pricing structures, we encourage submitting cost information:

1. Calculate average cost per trajectory for each domain
2. Include costs in USD using the optional `cost` field in your results
3. Document your cost calculation method in `methodology.notes` if using custom cost tracking

## Step 2: Prepare Submission Package

Use the CLI to prepare your submission automatically:

```bash
tau2 submit prepare \
  data/simulations/my_model_retail \
  data/simulations/my_model_airline \
  data/simulations/my_model_telecom \
  --output ./my_submission
```

This will:
- Find `results.json` in each simulation directory
- Verify all trajectory files are valid
- Check that submission requirements are met
- Compute performance metrics (Pass^k rates)
- Prompt for required metadata (model name, organization, contact email)
- Create a structured submission directory with:
  - `submission.json` — metadata and metrics
  - `trajectories/` — your trajectory files (to be shared externally, not committed to the repo)

To skip verification (e.g., for faster iteration):

```bash
tau2 submit prepare ... --output ./my_submission --no-verify
```

## Step 3: Validate Your Submission

```bash
tau2 submit validate ./my_submission
```

This verifies:
- All required files are present
- Trajectory files are valid
- Domain coverage is complete
- Model configurations are consistent
- Metrics match trajectory data

You can also verify individual trajectory files without a full submission:

```bash
tau2 submit verify-trajs \
  data/simulations/my_model_retail \
  data/simulations/my_model_airline \
  data/simulations/my_model_telecom
```

## Step 4: Create Your Submission Directory

1. Create a directory under `web/leaderboard/public/submissions/`
2. Name your directory using the format: `{model_name}_{model_organization}_{submission_date}`
   - Use lowercase letters, numbers, hyphens, and underscores only
   - Examples: `gpt-4.1_openai_2025-01-15`, `custom-model-v1_mycompany_2025-01-20`
3. Add only the `submission.json` file to the directory

Directory structure in the repo:

```
web/leaderboard/public/submissions/my-model_myorg_2025-01-15/
└── submission.json
```

> **Note:** Trajectory files are **not** committed to the repo — they are hosted on S3. Upload your trajectory files to an external service (Google Drive, HuggingFace, institutional storage, etc.) and include the download link in your PR description. A maintainer will upload them to S3 after review. Your shared trajectory files should follow this structure:
>
> **Text submissions:**
> ```
> trajectories/
> ├── my-model_airline_default_gpt-4o_4trials.json
> ├── my-model_retail_default_gpt-4o_4trials.json
> ├── my-model_telecom_default_gpt-4o_4trials.json
> └── my-model_banking_knowledge_gpt-4o_4trials.json
> ```
>
> **Voice submissions** (directory-based format, one experiment directory per domain):
> ```
> trajectories/
> └── <experiment_name>/
>     ├── results.json       # Metadata only
>     ├── simulations/       # Individual simulation data
>     └── artifacts/         # Audio files
> ```
>
> Keep the original structure as generated by `tau2 submit prepare`.

## Step 5: Update the Manifest

Add your directory name to the `submissions` array in `web/leaderboard/public/submissions/manifest.json`:

```json
{
  "submissions": [
    "existing-submission_org_2024-12-01",
    "my-model_myorg_2025-01-15"
  ],
  "legacy_submissions": [
    "older-model_org_2024-06-20"
  ]
}
```

| Array | Purpose | Leaderboard Display |
|-------|---------|---------------------|
| `submissions` | Current text submissions on the latest τ-bench version | Displayed normally |
| `legacy_submissions` | Older submissions from previous benchmark versions | Dimmed with "v1" badge, hidden by default |

> **Note for maintainers:** When a new benchmark version is released, move existing `submissions` entries to `legacy_submissions`.

## Step 6: Submit Pull Request

1. Fork the [τ-bench repository](https://github.com/sierra-research/tau2-bench)
2. Add your submission directory (with `submission.json` only) and update `manifest.json`
3. Submit a pull request with:
   - Clear description of your model and results
   - A link to your trajectory files hosted externally (Google Drive, HuggingFace, etc.)
   - Documentation of any framework modifications or task omissions
   - Link to your model/paper if available

> **Note:** After your PR is merged, submission metadata is automatically synced to S3 via a GitHub Actions workflow. Trajectory files are uploaded to S3 separately by a maintainer from the link you provide. The production website at [taubench.com](https://taubench.com) fetches all data from S3.

## JSON Schema Reference

Your `submission.json` must follow the schema defined in [`web/leaderboard/public/submissions/schema.json`](../web/leaderboard/public/submissions/schema.json).

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | string | Name of the model being evaluated |
| `model_organization` | string | Organization that developed the model |
| `submitting_organization` | string | Organization that ran the evaluation |
| `submission_date` | string | Date of submission (YYYY-MM-DD) |
| `contact_info` | object | Contact information (`email`, `name`, `github`) |
| `results` | object | Performance results per domain |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `submission_type` | string | `"standard"` | `"standard"` or `"custom"` |
| `modality` | string | `"text"` | `"text"` or `"voice"` |
| `is_new` | boolean | `false` | Highlight as new on the leaderboard |
| `trajectories_available` | boolean | `false` | Whether trajectory files are available on S3 |
| `trajectory_files` | object | — | Mapping of domain name to trajectory filename (on S3) |
| `references` | array | — | Links to papers, documentation, repos |
| `methodology` | object | — | Evaluation methodology details |
| `voice_config` | object | — | Voice-specific configuration (required for voice) |

### Domain Results

Each domain in `results` accepts:
- `pass_1` through `pass_4`: Success rate percentage (0-100) or `null`
- `cost`: Average cost in USD per trajectory or `null`
- `retrieval_config`: Required for `banking_knowledge` only

### References

The optional `references` array links to papers, blog posts, documentation, or other resources. Supported `type` values: `paper`, `blog_post`, `documentation`, `model_card`, `github`, `huggingface`, `other`.

## Verification System

Submissions are classified as **Verified** or **Unverified** on the leaderboard:

**Verified submissions** require:
- Trajectory data available for review (`trajectories_available: true`)
- No modifications to standard prompts (`modified_prompts: false`)
- Complete evaluation with no omitted tasks (`omitted_questions: false`)

**Unverified submissions** are marked with a caution icon and may have:
- Missing trajectory data
- Modified prompts or evaluation setup
- Omitted questions or domains

### Verification Fields

Include a `verification` section in the `methodology` object:

```json
"methodology": {
  "verification": {
    "modified_prompts": false,
    "omitted_questions": false,
    "details": "Full evaluation with standard configuration"
  }
}
```

- `modified_prompts` (boolean or null): `true` if prompts were modified, `false` if standard, `null` if unknown
- `omitted_questions` (boolean or null): `true` if tasks were omitted, `false` if all tasks evaluated, `null` if unknown
- `details` (string, optional): Additional context about the evaluation methodology

## Examples

### Standard Text Submission

```json
{
  "model_name": "My-Model-v1.0",
  "model_organization": "My Company",
  "submitting_organization": "My Company",
  "submission_date": "2025-01-15",
  "contact_info": {
    "email": "contact@mycompany.com",
    "name": "Research Team"
  },
  "is_new": true,
  "trajectories_available": true,
  "trajectory_files": {
    "retail": "my-model_retail_default_gpt-4o_4trials.json",
    "airline": "my-model_airline_default_gpt-4o_4trials.json",
    "telecom": "my-model_telecom_default_gpt-4o_4trials.json",
    "banking_knowledge": "my-model_banking_knowledge_gpt-4o_4trials.json"
  },
  "results": {
    "retail": { "pass_1": 75.2, "pass_2": 68.1, "pass_3": 62.3, "pass_4": 57.8, "cost": 0.15 },
    "airline": { "pass_1": 65.4, "pass_2": 60.1, "pass_3": 56.2, "pass_4": 53.0, "cost": 0.12 },
    "telecom": { "pass_1": 58.9, "pass_2": 52.1, "pass_3": 47.6, "pass_4": 43.2, "cost": 0.18 },
    "banking_knowledge": { "pass_1": 22.5, "pass_2": 17.3, "pass_3": 13.1, "pass_4": 10.2, "cost": 1.05, "retrieval_config": "terminal_use" }
  },
  "methodology": {
    "evaluation_date": "2025-01-10",
    "tau2_bench_version": "v1.0",
    "user_simulator": "gpt-4.1-2025-04-14",
    "notes": "Evaluated using default settings with 4 trials per task.",
    "verification": {
      "modified_prompts": false,
      "omitted_questions": false,
      "details": "Full evaluation with standard τ-bench configuration."
    }
  }
}
```

### Custom Submission (Modified Scaffold)

```json
{
  "model_name": "Custom-Multi-Agent-v1",
  "model_organization": "Research Lab",
  "submitting_organization": "Research Lab",
  "submission_date": "2025-01-15",
  "submission_type": "custom",
  "contact_info": { "email": "research@example.com" },
  "is_new": true,
  "trajectories_available": true,
  "references": [
    { "title": "Custom Agent Implementation", "url": "https://github.com/example/custom-tau-agent", "type": "github" }
  ],
  "results": {
    "retail": { "pass_1": 82.5, "pass_2": 78.1, "pass_3": 74.2, "pass_4": 71.0 },
    "airline": { "pass_1": 68.3, "pass_2": 63.5, "pass_3": 59.8, "pass_4": 56.2 },
    "telecom": { "pass_1": 75.1, "pass_2": 70.4, "pass_3": 66.8, "pass_4": 63.5 }
  },
  "methodology": {
    "evaluation_date": "2025-01-10",
    "tau2_bench_version": "0.2.0",
    "user_simulator": "gpt-4o",
    "notes": "Multi-model router with GPT-4 as planner and Claude-3.5-Sonnet as executor. Custom reflection step after each tool call.",
    "verification": {
      "modified_prompts": true,
      "omitted_questions": false,
      "details": "Custom system prompts for planning, execution, and reflection phases."
    }
  }
}
```

See `web/leaderboard/public/submissions/A_EXAMPLE_new-model_example-org_2025-01-15/` for a complete example.

---

## Voice Submissions (τ-voice)

Voice submissions evaluate audio-native models using full-duplex (simultaneous) communication. The voice evaluation stack is fundamentally different from text and requires significantly more infrastructure, so **voice submissions are handled on a case-by-case basis**.

### Why Voice Is Different

The voice user simulator is a complex multi-component system — LLM, TTS (via ElevenLabs), transcription (via Deepgram), audio effects pipeline, and decision models — that requires specific API keys and infrastructure to run. It is versioned separately via `VOICE_USER_SIMULATOR_VERSION` in `src/tau2/config.py`, with each version anchored to a git tag (`voice-user-sim-<version>`) for reproducibility.

Because of this complexity, we recommend that you **open a PR and contact us** so we can coordinate running the evaluation.

### Existing Provider (Adapter Already Integrated)

OpenAI, Gemini, and xAI already have audio-native adapters in `src/tau2/voice/audio_native/`. If you want results for one of these providers:

1. Open a PR with your `submission.json` and contact us — we can run the evaluation
2. If you ran the evaluation yourself, include a link to your trajectory data in the PR description for verification

The evaluation command (requires the full voice simulator stack):

```bash
tau2 run --domain retail --audio-native --audio-native-provider openai --audio-native-model gpt-realtime-1.5 --num-tasks 1 --verbose-logs
```

### New Provider (No Adapter Yet)

If the provider you want to evaluate doesn't have an adapter:

1. Implement an audio-native provider adapter integrating the provider's real-time WebSocket/audio API (see existing providers in `src/tau2/voice/audio_native/` for reference)
2. Open a PR with the adapter implementation and documentation
3. Contact us to coordinate running the evaluation

### Voice Trajectory Data

Voice trajectories contain audio data and tick-level simulation data, making them too large to include in the repository. Voice results use a directory-based storage format:

```
<experiment>/
├── results.json         # Metadata and task definitions only
├── simulations/         # Individual simulation data files
│   ├── sim_0.json
│   └── ...
└── artifacts/           # Runtime artifacts
    └── task_<id>/
        └── sim_<uuid>/
            └── audio/   # Conversation audio
```

When preparing a submission with `tau2 submit prepare`, monolithic JSON results are automatically converted to this directory format. The `prepare_submission` script also trims the output to include only canonical simulation audio (skipping debug logs, retried simulations, etc.).

Share your trajectory data via an external link (Google Drive, HuggingFace, institutional storage, etc.) and include the link in your PR description. We will use these to verify your results.

### Voice Submission Requirements

- Set `"modality": "voice"`
- Include a `voice_config` object (see fields below)
- Report results from the **"regular" speech complexity** split only (not "control")
- Set `trajectories_available` to `false`
- Add your directory to the `voice_submissions` array in `manifest.json` (not `submissions`)
- Include a link to your trajectory data in the PR description
- Set `methodology.user_simulator` to the voice user simulator version used (e.g., `"v1.0"` — see git tag `voice-user-sim-v1.0`)

> **Note:** Voice submissions typically only report Pass^1 scores since multi-trial evaluation with audio-native models is expensive. Higher Pass^k values may be `null`.

### Voice Config Fields

Required when `modality` is `"voice"`:

| Field | Required | Description |
|-------|----------|-------------|
| `provider` | Yes | Audio-native provider (e.g. `"openai"`, `"gemini"`, `"xai"`) |
| `model` | Yes | Model identifier (e.g. `"gpt-realtime-1.5"`) |
| `tick_duration_seconds` | No | Duration of each simulation tick in seconds |
| `max_steps_seconds` | No | Maximum simulation duration in seconds |
| `user_tts_provider` | No | User simulator TTS provider/model (e.g. `"elevenlabs/eleven_v3"`) |

### Voice Submission Example

```json
{
  "model_name": "gpt-realtime-1.5",
  "model_organization": "OpenAI",
  "submitting_organization": "Sierra",
  "submission_date": "2026-03-11",
  "modality": "voice",
  "contact_info": { "email": "research@sierra.ai", "name": "Research Team" },
  "is_new": true,
  "trajectories_available": false,
  "results": {
    "retail": { "pass_1": 43.9 },
    "airline": { "pass_1": 40.0 },
    "telecom": { "pass_1": 21.1 }
  },
  "voice_config": {
    "provider": "openai",
    "model": "gpt-realtime-1.5",
    "tick_duration_seconds": 0.2,
    "max_steps_seconds": 600,
    "user_tts_provider": "elevenlabs/eleven_v3"
  },
  "methodology": {
    "evaluation_date": "2026-03-01",
    "tau2_bench_version": "v2.0",
    "user_simulator": "v1.0",
    "notes": "Full-duplex audio-native evaluation using regular speech complexity.",
    "verification": {
      "modified_prompts": false,
      "omitted_questions": false
    }
  }
}
```

See `web/leaderboard/public/submissions/A_EXAMPLE_voice-model_example-org_2026-03-11/` for a complete example.

### Voice Manifest

Voice submissions go in the `voice_submissions` array in `manifest.json` (separate from text):

```json
{
  "submissions": [ ... ],
  "voice_submissions": [
    "my-voice-model_myorg_2026-03-01"
  ],
  "legacy_submissions": [ ... ]
}
```

---

## For Maintainers: Review Checklist

### Text Submissions

- [ ] `submission.json` follows the [schema](../web/leaderboard/public/submissions/schema.json)
- [ ] Directory name follows convention (`{model_name}_{model_organization}_{date}`)
- [ ] `manifest.json` updated (`submissions` array)
- [ ] Contact info is provided
- [ ] `submission_type` is set correctly (`"standard"` or `"custom"`)
- [ ] **If custom:** detailed `methodology.notes` and `references` with implementation links
- [ ] **If custom:** `methodology.verification.modified_prompts` is set appropriately
- [ ] `trajectory_files` mapping matches actual filenames
- [ ] 4 trials per domain (check trajectory files)
- [ ] **If banking domain:** `retrieval_config` field present in `banking_knowledge` results
- [ ] Results verified against trajectory data
- [ ] Cost information is positive numbers or `null` (if provided)
- [ ] No duplicate submissions
- [ ] PR includes link to externally hosted trajectory files
- [ ] **After merge:** download trajectories and upload to S3: `aws s3 cp <local-trajectories>/ s3://sierra-tau-bench-public/submissions/<submission-dir>/trajectories/ --recursive --profile tau-bench-ci`

### Voice Submissions

- [ ] `submission.json` follows the [schema](../web/leaderboard/public/submissions/schema.json)
- [ ] `modality` is `"voice"`, `trajectories_available` is `false`
- [ ] `voice_config` includes `provider` and `model`
- [ ] `manifest.json` updated (`voice_submissions` array)
- [ ] `methodology.user_simulator` set to voice user sim version (e.g., `"v1.0"`)
- [ ] PR description includes link to externally hosted trajectory data
- [ ] **New provider:** PR includes audio-native adapter implementation and documentation
- [ ] Results use "regular" speech complexity only
- [ ] No duplicate submissions

## Questions?

If you have questions about submitting results, please:
1. Check the [τ-bench documentation](https://github.com/sierra-research/tau2-bench)
2. Open an issue in the repository
3. Contact us at the email provided in the main README
