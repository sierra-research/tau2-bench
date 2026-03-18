# τ-bench Leaderboard Submissions

This directory contains model evaluation results for the τ-bench leaderboard. Each submission is a directory containing a `submission.json` and (optionally) trajectory files.

## Submission Types: Standard vs Custom

The leaderboard supports two types of submissions:

| Type | Description | Requirements |
|------|-------------|--------------|
| **Standard** | Uses the default τ-bench scaffold (base LLM + standard tools/prompts) | Default - no extra documentation needed |
| **Custom** | Uses modified scaffolds (multi-model routers, additional tools, custom prompting, etc.) | Requires detailed methodology documentation |

### Standard Submissions
If you're running τ-bench as documented without modifications, your submission is **standard**. You can omit the `submission_type` field (it defaults to `"standard"`).

### Custom Submissions
If you made **any** modifications to the evaluation setup, you must:
1. Set `"submission_type": "custom"` in your JSON
2. Provide detailed `methodology.notes` explaining your modifications
3. Include a link to your implementation in `references`
4. Set `methodology.verification.modified_prompts` appropriately

See the [Custom Submissions](#custom-submission-requirements) section below for details.

## Submission Modalities: Text vs Voice

The leaderboard supports two evaluation modalities:

| Modality | Description | Communication Mode |
|----------|-------------|--------------------|
| **Text** (default) | Standard text-based half-duplex evaluation | Turn-based |
| **Voice** | Audio-native full-duplex evaluation (τ-voice) | Simultaneous/streaming |

### Text Submissions
If you ran a standard τ-bench text evaluation, your submission uses the default `"text"` modality. You can omit the `modality` field (it defaults to `"text"`).

### Voice Submissions
If you ran a τ-voice audio-native evaluation, you must:
1. Set `"modality": "voice"` in your JSON
2. Include a `voice_config` object with provider and model details
3. Report results from the **"regular" speech complexity** split only (not "control")
4. Trajectory files are **not included** for voice submissions (audio files are too large)

The `tau2 submit prepare` CLI command auto-detects voice mode from your input data.

## How to Submit Results

1. **Evaluate your model** using the [τ-bench framework](https://github.com/sierra-research/tau2-bench)
2. **Create a submission directory** with a `submission.json` file following the schema defined in `schema.json`
3. **Name your directory** using the format: `{model_name}_{model_organization}_{submission_date}`
   - Example: `gpt-4.1_openai_2025-01-15`
   - Use lowercase, replace spaces with hyphens
4. **Update the manifest file** by adding your directory name to the appropriate array in `manifest.json`: `submissions` for text or `voice_submissions` for voice (see [Manifest Structure](#manifest-structure) below)
5. **Submit a pull request** to this repository with both files added/modified
   - **Include trajectory links in your PR description** for verification (see below)
6. **Wait for review** - we'll validate your submission and merge if everything looks correct

## Directory Naming Convention

- Use lowercase letters, numbers, hyphens, and underscores only
- Format: `{model_name}_{model_organization}_{submission_date}`
- Each submission is a directory containing a `submission.json` file
- Examples:
  - `claude-3-7-sonnet_anthropic_2025-01-15`
  - `custom-model-v1_mycompany_2025-01-20`
  - `gpt-5_openai_2025-02-01`

## JSON Schema

Your submission must follow the schema defined in `schema.json`. Here's a quick example of a **standard** submission:

```json
{
  "model_name": "My-Amazing-Model-v1.0",
  "model_organization": "My Company",
  "submitting_organization": "My Company",
  "submission_date": "2025-01-15",
  "submission_type": "standard",
  "contact_info": {
    "email": "contact@mycompany.com",
    "name": "John Doe",
    "github": "johndoe"
  },
  "is_new": true,
  "trajectories_available": true,
  "references": [
    {
      "title": "Model Technical Paper",
      "url": "https://arxiv.org/abs/2401.00000",
      "type": "paper"
    },
    {
      "title": "Model Documentation",
      "url": "https://docs.example.com/model",
      "type": "documentation"
    }
  ],
  "results": {
    "retail": {
      "pass_1": 75.2,
      "pass_2": 68.1,
      "pass_3": 62.3,
      "pass_4": 57.8,
      "cost": 0.15
    },
    "airline": {
      "pass_1": 65.4,
      "pass_2": null,
      "pass_3": null,
      "pass_4": null,
      "cost": 0.12
    },
    "telecom": {
      "pass_1": 58.9,
      "pass_2": 52.1,
      "pass_3": 47.6,
      "pass_4": 43.2,
      "cost": 0.18
    },
    "banking_knowledge": {
      "pass_1": 22.5,
      "pass_2": 17.3,
      "pass_3": 13.1,
      "pass_4": 10.2,
      "cost": 1.05,
      "retrieval_config": "terminal_use"
    }
  },
  "methodology": {
    "evaluation_date": "2025-01-10",
    "tau2_bench_version": "v1.0",
    "user_simulator": "gpt-4.1-2025-04-14",
    "notes": "Evaluated using default settings with 4 trials per task",
    "verification": {
      "modified_prompts": false,
      "omitted_questions": false,
      "details": "Full evaluation with standard τ-bench configuration"
    }
  }
}
```

## Custom Submission Requirements

If your submission uses a custom scaffold (`"submission_type": "custom"`), you **must** provide detailed documentation to help others understand your approach.

### Required for Custom Submissions

1. **Set the submission type:**
   ```json
   "submission_type": "custom"
   ```

2. **Detailed methodology notes** - Explain what you modified and why:
   ```json
   "methodology": {
     "notes": "This submission uses a multi-model router that selects between GPT-4 and Claude based on task complexity. We added a custom reflection step after each tool call. The router is trained on a separate dataset of task classifications."
   }
   ```

3. **Link to implementation** - Include a reference to your code:
   ```json
   "references": [
     {
       "title": "Custom Agent Implementation",
       "url": "https://github.com/your-org/custom-tau-agent",
       "type": "github"
     }
   ]
   ```

4. **Set verification fields appropriately:**
   ```json
   "methodology": {
     "verification": {
       "modified_prompts": true,
       "omitted_questions": false,
       "details": "Modified agent system prompt to include reflection instructions."
     }
   }
   ```

### Examples of Custom Modifications

Custom submissions might include:
- Multi-model routers or model ensembles
- Additional tools beyond the standard τ-bench tool set
- Modified agent orchestration or control flow
- Any other modification to the default evaluation setup

### Complete Custom Submission Example

```json
{
  "model_name": "Custom-Multi-Agent-v1",
  "model_organization": "Research Lab",
  "submitting_organization": "Research Lab",
  "submission_date": "2025-01-15",
  "submission_type": "custom",
  "contact_info": {
    "email": "research@example.com"
  },
  "is_new": true,
  "trajectories_available": true,
  "references": [
    {
      "title": "Multi-Agent System Implementation",
      "url": "https://github.com/research-lab/multi-agent-tau",
      "type": "github"
    },
    {
      "title": "Technical Blog Post",
      "url": "https://blog.example.com/multi-agent-approach",
      "type": "blog_post"
    }
  ],
  "results": {
    "retail": { "pass_1": 82.5, "pass_2": 78.1, "pass_3": 74.2, "pass_4": 71.0 },
    "airline": { "pass_1": 68.3, "pass_2": 63.5, "pass_3": 59.8, "pass_4": 56.2 },
    "telecom": { "pass_1": 75.1, "pass_2": 70.4, "pass_3": 66.8, "pass_4": 63.5 },
    "banking_knowledge": { "pass_1": 22.5, "pass_2": 17.3, "pass_3": 13.1, "pass_4": 10.2, "retrieval_config": "terminal_use" }
  },
  "methodology": {
    "evaluation_date": "2025-01-10",
    "tau2_bench_version": "0.2.0",
    "user_simulator": "gpt-4o",
    "notes": "This system uses a planning-execution-reflection loop with GPT-4 as the planner and Claude-3.5-Sonnet as the executor. A separate classifier routes complex tasks to a specialized reasoning pipeline. See our GitHub repo for complete implementation.",
    "verification": {
      "modified_prompts": true,
      "omitted_questions": false,
      "details": "Custom system prompts for planning, execution, and reflection phases. All τ-bench tasks were evaluated without omission."
    }
  }
}
```

## Voice Submission

Voice submissions evaluate audio-native models using full-duplex (simultaneous) communication. The key differences from text submissions are:

- **`modality`** is set to `"voice"`
- **`voice_config`** provides audio-native configuration for reproducibility
- **`trajectories_available`** is `false` (audio trajectories are too large)
- **Results use only the "regular" speech complexity** split (realistic audio conditions)

### Voice Config Fields

| Field | Required | Description |
|-------|----------|-------------|
| `provider` | ✅ | Audio-native provider (e.g. `"openai"`, `"gemini"`, `"xai"`) |
| `model` | ✅ | Model identifier (e.g. `"gpt-realtime-1.5"`) |
| `tick_duration_seconds` | Optional | Duration of each simulation tick in seconds |
| `max_steps_seconds` | Optional | Maximum simulation duration in seconds |
| `user_tts_provider` | Optional | User simulator TTS provider/model (e.g. `"elevenlabs/eleven_v3"`) |

### Complete Voice Submission Example

```json
{
  "model_name": "gpt-realtime-1.5",
  "model_organization": "OpenAI",
  "submitting_organization": "Sierra",
  "submission_date": "2026-03-11",
  "submission_type": "standard",
  "modality": "voice",
  "contact_info": {
    "email": "research@sierra.ai",
    "name": "Research Team"
  },
  "is_new": true,
  "trajectories_available": false,
  "results": {
    "retail": {
      "pass_1": 43.9
    },
    "airline": {
      "pass_1": 40.0
    },
    "telecom": {
      "pass_1": 21.1
    }
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
    "user_simulator": "gpt-4.1-2025-04-14",
    "notes": "Full-duplex audio-native evaluation using regular speech complexity (realistic audio conditions).",
    "verification": {
      "modified_prompts": false,
      "omitted_questions": false
    }
  }
}
```

> **Note:** Voice submissions typically only report Pass^1 scores since multi-trial evaluation with audio-native models is expensive. Higher pass^k values may be `null`.

---

## Verification System

### What Makes a Submission "Verified"?

To ensure result authenticity and reproducibility, submissions are classified as **Verified** or **Unverified** based on these criteria:

**✅ Verified Submissions Must Have:**
- Full trajectory data available for review
- No modifications to standard user simulator or agent prompts
- Complete evaluation (no omitted questions/tasks from any domain)
- Clear methodology documentation

**⚠️ Unverified Submissions** are marked with a caution icon and may have:
- Missing trajectory data
- Unknown evaluation methodology
- Modified prompts or evaluation setup
- Incomplete evaluation (e.g., only Pass@1 scores)
- Omitted questions or entire domains

### Verification Fields

Your submission must include a `verification` section in the `methodology` object:

```json
"methodology": {
  "evaluation_date": "2025-01-10", 
  "tau2_bench_version": "v1.0",
  "user_simulator": "gpt-4.1-2025-04-14",
  "notes": "Additional methodology notes",
  "verification": {
    "modified_prompts": false,
    "omitted_questions": false,
    "details": "Any additional verification details"
  }
}
```

**Required verification fields:**
- `modified_prompts` (boolean): Set to `true` if you modified user simulator or agent prompts, `false` if using standard prompts, or `null` if unknown
- `omitted_questions` (boolean): Set to `true` if you omitted any questions/tasks, `false` if you evaluated all available tasks, or `null` if unknown  
- `details` (string, optional): Additional context about your evaluation methodology

### Why This Matters

The verification system helps researchers and practitioners:
- **Identify reliable results** for fair model comparisons
- **Understand evaluation differences** that might affect scores
- **Make informed decisions** about which results to trust
- **Maintain scientific rigor** in benchmark reporting

Unverified submissions are still valuable for providing initial performance estimates, but verified submissions offer higher confidence for research and decision-making.

## Banking Knowledge Domain

The `banking_knowledge` domain evaluates agents on knowledge-intensive customer support tasks over a fintech knowledge base. Because performance depends heavily on how agents access the knowledge base, submissions for this domain **must** include a `retrieval_config` field specifying the retrieval method used.

### Retrieval Config Field

Add `retrieval_config` inside the `banking_knowledge` results object:

```json
"banking_knowledge": {
  "pass_1": 22.5,
  "pass_2": 17.3,
  "pass_3": 13.1,
  "pass_4": 10.2,
  "cost": 1.05,
  "retrieval_config": "terminal_use"
}
```

### Supported Retrieval Configurations

The `retrieval_config` value is a free-form descriptive label (not necessarily the `--retrieval-config` CLI flag value). Common values:

| Config | Description |
|--------|-------------|
| `terminal_use` | Agent navigates the knowledge base via shell commands (grep, cat, find, etc.) |
| `openai_embeddings` | Dense retrieval using OpenAI's text-embedding-3-large model |
| `qwen_embeddings` | Dense retrieval using Qwen3-Embedding model |
| `bm25` | Sparse retrieval using BM25 |

The `retrieval_config` value is displayed on the leaderboard as a badge in the banking knowledge domain view, making it easy to compare results across different retrieval strategies.

> **Note:** If you use a retrieval configuration not listed above, use a short descriptive string (e.g., `"custom_reranker"`) and document it in your `methodology.notes`.

## Important Notes

- **Pass@k scores**: Use `null` for missing data (e.g., if you only evaluated Pass@1)
- **Trajectories**: Set `trajectories_available: true` if you're including trajectory files in your submission, `false` otherwise. Voice submissions should always set this to `false`
- **Modality**: Defaults to `"text"` if omitted. Set to `"voice"` for audio-native evaluations
- **Voice config**: Required when `modality` is `"voice"`. Must include `provider` and `model` at minimum
- **References**: Include links to papers, documentation, or other resources about your model (optional but recommended)
- **Cost information**: Optionally include the average cost in USD to run one trajectory in each domain using the `cost` field. This helps with fair comparisons between models with different pricing structures
- **Domains**: You don't need to evaluate all domains, but include at least one
- **Manifest file**: Don't forget to add your directory name to the appropriate array in `manifest.json` — `submissions` for text, `voice_submissions` for voice. The leaderboard loads submissions from these lists (see [Manifest Structure](#manifest-structure))
- **Trajectory verification**: Include a link to your evaluation trajectories in your pull request description
- **Verification**: We will verify results using the provided trajectories before publishing
- **Updates**: If you need to update your submission, submit a new PR with the corrected file

## References Field

The optional `references` field allows you to include links to papers, blog posts, documentation, or other resources about your model. This helps researchers and practitioners learn more about your model.

### Reference Format

```json
"references": [
  {
    "title": "Paper Title or Description",
    "url": "https://example.com/link",
    "type": "paper"
  },
  {
    "title": "Model Documentation",
    "url": "https://docs.example.com",
    "type": "documentation"
  }
]
```

### Supported Reference Types

- `paper`: Research papers (arXiv, conference proceedings, etc.)
- `blog_post`: Blog posts or technical articles
- `documentation`: Official documentation or API guides
- `model_card`: Model cards or technical specifications
- `github`: GitHub repositories
- `huggingface`: Hugging Face model pages
- `other`: Any other type of reference

### Examples

```json
"references": [
  {
    "title": "GPT-4 Technical Report",
    "url": "https://arxiv.org/abs/2303.08774",
    "type": "paper"
  },
  {
    "title": "Claude 3 Model Card",
    "url": "https://www.anthropic.com/claude",
    "type": "model_card"
  },
  {
    "title": "Model GitHub Repository",
    "url": "https://github.com/org/model",
    "type": "github"
  }
]
```

## Manifest Structure

The `manifest.json` file controls which submissions appear on the leaderboard. It contains three arrays:

```json
{
  "submissions": [
    "my-model_mycompany_2025-01-15"
  ],
  "voice_submissions": [
    "my-voice-model_mycompany_2026-03-01"
  ],
  "legacy_submissions": [
    "old-model_othercompany_2024-06-20"
  ]
}
```

| Array | Description | Leaderboard Display |
|-------|-------------|---------------------|
| `submissions` | Current text submissions evaluated on the latest version of τ-bench | Displayed normally |
| `voice_submissions` | Current voice (audio-native) submissions | Displayed in the voice leaderboard tab |
| `legacy_submissions` | Older submissions evaluated on a previous version of the benchmark | Dimmed with a "v1" badge; hidden by default behind a toggle |

### When to use each array

- **`submissions`**: Add your directory here for text-modality evaluations against the current version of τ-bench. New text submissions should always go here.
- **`voice_submissions`**: Add your directory here for voice-modality (audio-native) evaluations. Set `"modality": "voice"` in your `submission.json`.
- **`legacy_submissions`**: Submissions from earlier benchmark versions are moved here when the benchmark is updated. These remain visible on the leaderboard for historical reference but are visually distinguished from current results.

> **Note for maintainers:** When a new benchmark version is released, move all existing entries from `submissions` to `legacy_submissions` and update the `last_updated` timestamp.

---

## Validation

Your JSON file will be automatically validated against the schema when you submit a pull request. Make sure:

- All required fields are present
- Email format is valid
- Scores are between 0-100 or `null`
- Cost values are positive numbers or `null` (if provided)
- Date format is YYYY-MM-DD
- **`submission_type`** is `"standard"` or `"custom"` (defaults to `"standard"` if omitted)
- **If custom:** detailed methodology notes and implementation links are provided
- **Verification fields are included** in the methodology section
- **Trajectory link is provided** in the pull request description (for verified submissions)

### Verification Requirements

For **verified submissions**:
- ✅ Include trajectory data link in your PR description
- ✅ Set `modified_prompts: false` if using standard prompts
- ✅ Set `omitted_questions: false` if evaluating all available tasks
- ✅ Set `trajectories_available: true` in the root submission object

For **unverified submissions**:
- ⚠️ Use `null` values for unknown verification fields
- ⚠️ Set `trajectories_available: false`
- ⚠️ Clearly document limitations in the `details` field

## Trajectory Requirements

When submitting your pull request, please include a link to your evaluation trajectories in the PR description. These are needed to verify your results and ensure reproducibility.

### What to Include
Your trajectory data should contain the raw evaluation output from τ-bench, including:
- All conversation logs between your model and the user simulator
- Action sequences and tool calls made by your model
- Task completion status for each evaluation run
- Any relevant configuration or setup details

### Trajectory File Setup
Place your trajectory files inside a `trajectories/` subdirectory in your submission folder. **Keep the original filenames** as generated by τ-bench:

```
your-submission-dir/
├── submission.json
└── trajectories/
    ├── my-model_airline_default_gpt-4o_4trials.json
    ├── my-model_retail_default_gpt-4o_4trials.json
    ├── my-model_telecom_default_gpt-4o_4trials.json
    └── my-model_banking_knowledge_gpt-4o_4trials.json
```

Then add a `trajectory_files` mapping in your `submission.json` so the leaderboard knows which file corresponds to which domain:

```json
{
  "trajectories_available": true,
  "trajectory_files": {
    "airline": "my-model_airline_default_gpt-4o_4trials.json",
    "retail": "my-model_retail_default_gpt-4o_4trials.json",
    "telecom": "my-model_telecom_default_gpt-4o_4trials.json",
    "banking_knowledge": "my-model_banking_knowledge_gpt-4o_4trials.json"
  }
}
```

### Where to Host
We recommend hosting trajectory files on:
- **GitHub Releases** (preferred for open models)
- **HuggingFace model repositories** 
- **Google Drive** (with public sharing enabled)
- **Institutional repositories** with public access
- **Other cloud storage** with public download links

### Acceptable Formats
- ZIP archives
- TAR.GZ compressed files
- Direct links to GitHub/HuggingFace repositories
- Any format that allows reviewers to access the raw trajectory data

## Questions?

If you have questions about submitting results, please:
1. Check the [τ-bench documentation](https://github.com/sierra-research/tau2-bench)
2. Open an issue in this repository
3. Contact us at the email provided in the main README

Thank you for contributing to the τ-bench leaderboard!
