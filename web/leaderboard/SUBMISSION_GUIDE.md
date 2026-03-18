# τ-bench Leaderboard Submission Guide

This repository supports community submissions to the leaderboard through pull requests! Here's how it works:

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

**Other Modifications:**
- Any other modifications to the default evaluation setup

**⚠️ Requirements for Custom Submissions:**

Custom submissions **must** include detailed methodology documentation:

1. **Set `submission_type` to `"custom"`** in your submission.json

2. **Provide comprehensive `methodology.notes`** explaining:
   - What modifications were made to the standard scaffold
   - If domain-trained: describe the training approach, data sources, and any overlap with τ-bench domains or tasks
   - Why these modifications were made
   - How the custom system works at a high level

3. **Link to your implementation** in the `references` array:
   - Include a GitHub link to your code/fork
   - Link to the paper describing your training methodology (if applicable)
   - Provide documentation or a blog post if available

4. **Set `methodology.verification.modified_prompts` to `true`** if you modified any prompts

**Example 1:** Custom scaffold with modified prompts
```json
{
  "submission_type": "custom",
  "methodology": {
    "evaluation_date": "2025-01-15",
    "tau2_bench_version": "0.2.0",
    "user_simulator": "gpt-4o",
    "notes": "This submission uses a multi-model router that selects between GPT-4 and Claude based on task complexity. We also added a custom reflection step after each tool call. See our GitHub repo for full implementation details.",
    "verification": {
      "modified_prompts": true,
      "omitted_questions": false,
      "details": "Modified the agent system prompt to include reflection instructions. No questions were omitted."
    }
  },
  "references": [
    {
      "title": "Our Custom Agent Implementation",
      "url": "https://github.com/example/custom-tau-agent",
      "type": "github"
    }
  ]
}
```

**Example 2:** Domain-trained model
```json
{
  "submission_type": "custom",
  "methodology": {
    "evaluation_date": "2025-01-15",
    "tau2_bench_version": "0.2.0",
    "user_simulator": "gpt-4o",
    "notes": "This model was post-trained using RL on customer service scenarios including airline and retail domains. Training data was synthetically generated based on similar task distributions. The model uses the standard τ-bench scaffold without prompt modifications.",
    "verification": {
      "modified_prompts": false,
      "omitted_questions": false,
      "details": "Standard evaluation protocol. Model was trained on data overlapping with τ-bench domains."
    }
  },
  "references": [
    {
      "title": "Our Training Methodology Paper",
      "url": "https://arxiv.org/abs/xxxx.xxxxx",
      "type": "paper"
    },
    {
      "title": "Model Weights on HuggingFace",
      "url": "https://huggingface.co/example/our-agent-model",
      "type": "huggingface"
    }
  ]
}
```

---

## How to Submit Results

### Step 1: Evaluate Your Model and Generate Trajectories
Use the [τ-bench framework](https://github.com/sierra-research/tau-bench) to evaluate your model on one or more domains (retail, airline, telecom, banking). This process will generate both your performance metrics and the trajectory files required for submission.

#### How to Run τ-bench Evaluation
1. Clone and set up the [τ-bench repository](https://github.com/sierra-research/tau-bench):
   ```bash
   git clone https://github.com/sierra-research/tau-bench
   cd tau-bench
   uv sync
   ```

2. Set up your LLM API keys by copying `.env.example` to `.env` and adding your API keys.

3. Run evaluations on all domains for complete leaderboard submission:
   ```bash
   # Airline domain (minimum 4 trials required)
   tau2 run --domain airline --agent-llm your-model-name --user-llm your-user-sim-llm --num-trials 4
   
   # Retail domain (minimum 4 trials required)
   tau2 run --domain retail --agent-llm your-model-name --user-llm your-user-sim-llm --num-trials 4
   
   # Telecom domain (minimum 4 trials required)
   tau2 run --domain telecom --agent-llm your-model-name --user-llm your-user-sim-llm --num-trials 4
   
   # Optional: Run with more trials for better statistics
   # tau2 run --domain airline --agent-llm your-model-name --user-llm your-user-sim-llm --num-trials 8
   ```
   
   **Important Notes:**
   - You can use any LLM as the user simulator, but this choice will be reported on the leaderboard
   - **We strongly prefer results with 4+ trials** 

4. **Banking domain (τ-Knowledge):** The banking domain requires a retrieval configuration for the knowledge base. You must include a `retrieval_config` field in your `banking_knowledge` results specifying which retrieval method was used. Supported values are `terminal`, `text-emb-3-large`, `qwen3-emb`, and `bm25`. If using a different method, use a short descriptive string and document it in `methodology.notes`.
   ```bash
   # Banking domain with terminal-based retrieval
   tau2 run --domain banking_knowledge --retrieval-config terminal_use --agent-llm your-model-name --user-llm your-user-sim-llm --num-trials 4
   
   # Banking domain with BM25 retrieval
   tau2 run --domain banking_knowledge --retrieval-config bm25 --agent-llm your-model-name --user-llm your-user-sim-llm --num-trials 4
   ```

5. Your trajectory files will be saved in `data/tau2/simulations/` within the τ-bench directory.

#### Cost Tracking (Optional but Recommended)
To enable fair comparisons between models with different pricing structures, we encourage submitting cost information:

1. **Calculate average cost per trajectory** for each domain by dividing total cost by number of trajectories run
2. **Include costs in USD** when creating your submission file using the optional `cost` field
3. **Document cost calculation method** in the `methodology.notes` field if using custom cost tracking

### Step 2: Document Any Framework Modifications or Task Omissions
If you made any changes to the τ-bench framework or evaluation protocol, you **must** document these in your pull request:

#### Framework Modifications
If you modified the τ-bench framework (user simulation, prompts, evaluation logic, etc.):
1. **Fork Documentation**: Provide a link to your τ-bench fork if you made any code changes
2. **Change Summary**: Clearly describe what was modified and why in your pull request description
3. **Code References**: Link to specific commits, files, or line numbers where changes were made
4. **Reproducibility**: Ensure others can reproduce your results using your modified framework

#### Task Omissions
If you omitted any tasks from your evaluation runs:
1. **List Omitted Tasks**: Specify which tasks were skipped (by task ID or description)
2. **Reason for Omission**: Clearly explain why these tasks were omitted (e.g., technical limitations, resource constraints, model capabilities)
3. **Impact Assessment**: Describe how this might affect the interpretation of your results

**Important**: We strongly prefer submissions that link to τ-bench forks rather than describing changes in text, as this ensures full transparency and reproducibility.

### Step 3: Create Your Submission Directory
1. Navigate to the `public/submissions/` directory
2. Create a new directory following the naming convention: `{model-name}_{model_organization}_{date}`
3. Inside your submission directory, create:
   - `submission.json` - Your submission metadata (using schema defined in `public/submissions/schema.json`)
   - `trajectories/` directory - For your trajectory files

**Important:** If you made any modifications to the standard τ-bench scaffold, set `"submission_type": "custom"` and provide detailed methodology documentation. See [Submission Types](#submission-types-standard-vs-custom) above.

Example directory structure:
```
public/submissions/my-awesome-model_mycompany_2025-01-15/
├── submission.json
└── trajectories/
    ├── my-awesome-model_airline_default_gpt-4o_4trials.json
    ├── my-awesome-model_retail_default_gpt-4o_4trials.json
    ├── my-awesome-model_telecom_default_gpt-4o_4trials.json
    └── my-awesome-model_banking_knowledge_gpt-4o_4trials.json
```

### Step 4: Add Your Trajectory Files
1. Copy your trajectory files from `data/tau2/simulations/` to your submission's `trajectories/` directory
2. **Keep the original filenames** as generated by τ-bench (they contain useful metadata like model name, domain, user simulator, and trial count)
3. Add trajectory files for all domains you evaluated
4. **Set `trajectories_available` to `true`** in your `submission.json`
5. **Add a `trajectory_files` mapping** in your `submission.json` so the leaderboard knows which file corresponds to which domain:
   ```json
   {
     "trajectories_available": true,
     "trajectory_files": {
       "airline": "my-awesome-model_airline_default_gpt-4o_4trials.json",
       "retail": "my-awesome-model_retail_default_gpt-4o_4trials.json",
       "telecom": "my-awesome-model_telecom_default_gpt-4o_4trials.json",
       "banking_knowledge": "my-awesome-model_banking_knowledge_gpt-4o_4trials.json"
     }
   }
   ```
6. **Include `methodology.verification`** to indicate whether prompts were modified or questions omitted:
   ```json
   {
     "methodology": {
       "user_simulator": "gpt-4o",
       "notes": "Banking domain evaluated with terminal-based retrieval.",
       "verification": {
         "modified_prompts": false,
         "omitted_questions": false
       }
     }
   }
   ```
   This is required for your submission to show as "✅ Verified" on the leaderboard.

7. **For banking domain submissions**, include the `retrieval_config` field in your `banking_knowledge` results:
   ```json
   {
     "results": {
       "banking_knowledge": {
         "pass_1": 22.5,
         "pass_2": 17.3,
         "pass_3": 13.1,
         "pass_4": 10.2,
         "retrieval_config": "terminal"
       }
     }
   }
   ```
   Supported values: `terminal`, `text-emb-3-large`, `qwen3-emb`, `bm25`. This is displayed as a badge on the leaderboard.

### Step 5: Update the Manifest
Add your directory name to the `submissions` array in `public/submissions/manifest.json`:

```json
{
  "submissions": [
    "existing-submission-1_org_2024-12-01",
    "existing-submission-2_org_2024-12-15", 
    "my-awesome-model_mycompany_2025-01-15"
  ],
  "legacy_submissions": [
    "older-model_org_2024-06-20"
  ]
}
```

**Important:** The manifest has two arrays:

| Array | Purpose |
|-------|---------|
| `submissions` | **Current** submissions evaluated on the latest τ-bench version. **Add new submissions here.** |
| `legacy_submissions` | **Older** submissions from previous benchmark versions. Displayed dimmed with a "v1" badge on the leaderboard and hidden by default behind a toggle. |

> When a new benchmark version is released, maintainers move existing `submissions` entries to `legacy_submissions`.

### Step 6: Submit Pull Request
1. Fork the [τ-bench repository](https://github.com/sierra-research/tau-bench)
2. Add your submission directory with submission.json, trajectory files, and update the manifest in the `web/leaderboard/` directory
3. Submit a pull request with:
   - Clear description of your model and results
   - **REQUIRED: Complete submission directory with trajectory files in `trajectories/` subdirectory**
   - Complete trajectory files for all evaluated domains (airline, retail, telecom, banking_knowledge)
   - **Documentation of any τ-bench framework modifications or task omissions** (link to your fork, describe changes/omissions)
   - Link to your model/paper if available
   - Contact information for questions

### Step 7: Review Process
We'll review your submission for:
- Schema compliance
- Data consistency
- Proper formatting

## For Maintainers: Processing Submissions

### Automatic Loading
The leaderboard automatically loads all submissions listed in `public/submissions/manifest.json`. No code changes needed for new submissions!

### Review Checklist
- [ ] submission.json follows the schema
- [ ] Directory name follows convention
- [ ] Manifest.json is updated
- [ ] Contact info is provided
- [ ] **`submission_type` is set correctly** (`"standard"` or `"custom"`)
- [ ] **If custom:** detailed `methodology.notes` explaining modifications
- [ ] **If custom:** `references` includes link to implementation/code
- [ ] **If custom:** `methodology.verification.modified_prompts` is set appropriately
- [ ] Framework modifications and task omissions documented in PR description (if any changes or omissions)
- [ ] **Trajectory files uploaded to submission's `trajectories/` directory**
- [ ] `trajectory_files` mapping in submission.json matches actual filenames
- [ ] Complete trajectory files for all evaluated domains
- [ ] **4 trials per domain** (check num-trials in trajectory files)
- [ ] **If banking domain:** `retrieval_config` field included in `banking_knowledge` results
- [ ] **If banking domain:** retrieval configuration documented in `methodology.notes`
- [ ] Results verified against trajectory data
- [ ] Trajectory files contain valid τ-bench output format
- [ ] Cost information is positive numbers or null (if provided)
- [ ] No duplicate submissions

### File Structure
```
public/submissions/
├── README.md                           # Contributor guidelines
├── schema.json                         # JSON schema definition
├── manifest.json                       # List of active submissions
├── model1_org1_date/                   # Individual submission directory
│   ├── submission.json                 # Submission metadata
│   └── trajectories/                   # Trajectory files for this submission
│       ├── model1_airline_default_gpt-4o_4trials.json
│       ├── model1_retail_default_gpt-4o_4trials.json
│       ├── model1_telecom_default_gpt-4o_4trials.json
│       └── model1_banking_knowledge_gpt-4o_4trials.json
├── model2_org2_date/                   # Another submission
│   ├── submission.json
│   └── trajectories/
│       ├── model2_airline_default_claude-3-5-sonnet_4trials.json
│       └── ...
└── ...
```

## Example Submission

See the directory `public/submissions/A_EXAMPLE_new-model_example-org_2025-01-15/` for a complete example of what a submission should look like.

### Example Directory Structure
```
public/submissions/my-awesome-model_mycompany_2025-01-15/
├── submission.json
└── trajectories/
    ├── my-awesome-model_airline_default_gpt-4o_4trials.json
    ├── my-awesome-model_retail_default_gpt-4o_4trials.json
    ├── my-awesome-model_telecom_default_gpt-4o_4trials.json
    └── my-awesome-model_banking_knowledge_gpt-4o_4trials.json
```

### Upload Instructions
1. Copy trajectory files from your τ-bench `data/tau2/simulations/` directory
2. Keep the original filenames — no renaming needed
3. Add them to your submission's `trajectories/` directory in your pull request
4. Add the `trajectory_files` mapping to your `submission.json`
5. Do not compress or archive the files — upload the raw JSON files

### Generating Complete Trajectories
Refer to [Step 1: Evaluate Your Model and Generate Trajectories](#step-1-evaluate-your-model-and-generate-trajectories) for detailed instructions on running τ-bench evaluations.

## Technical Details

The leaderboard component (`src/components/Leaderboard.jsx`) automatically:
1. Fetches the manifest file on page load
2. Loads each submission file listed in the manifest
3. Converts the JSON format to the internal data structure
4. Renders the leaderboard with all submissions
