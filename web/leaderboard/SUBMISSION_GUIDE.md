# Tau2-Bench Leaderboard Submission Guide

This repository supports community submissions to the leaderboard through pull requests! Here's how it works:

## Submission Types: Standard vs Custom

The leaderboard distinguishes between two types of submissions:

### Standard Submissions (Default)
Standard submissions use the **default τ-bench scaffold**:
- A base LLM as the agent
- The standard tool set provided by τ-bench
- Default prompts and evaluation protocol

If you're running τ-bench as documented without modifications, your submission is **standard**. You don't need to specify `submission_type` in your JSON (it defaults to `"standard"`).

### Custom Submissions
Custom submissions use **modified scaffolds or approaches**, such as:
- Multi-model routers or model ensembles
- Additional tools beyond the standard τ-bench tool set
- Modified agent orchestration or control flow
- Any other modifications to the default evaluation setup

**⚠️ Requirements for Custom Submissions:**

Custom submissions **must** include detailed methodology documentation:

1. **Set `submission_type` to `"custom"`** in your submission.json

2. **Provide comprehensive `methodology.notes`** explaining:
   - What modifications were made to the standard scaffold
   - Why these modifications were made
   - How the custom system works at a high level

3. **Link to your implementation** in the `references` array:
   - Include a GitHub link to your code/fork
   - Provide documentation or a blog post if available

4. **Set `methodology.verification.modified_prompts` to `true`** if you modified any prompts

Example methodology section for a custom submission:
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

---

## How to Submit Results

### Step 1: Evaluate Your Model and Generate Trajectories
Use the [tau2-bench framework](https://github.com/sierra-research/tau2-bench) to evaluate your model on one or more domains (retail, airline, telecom). This process will generate both your performance metrics and the trajectory files required for submission.

#### How to Run tau2-bench Evaluation
1. Clone and set up the [tau2-bench repository](https://github.com/sierra-research/tau2-bench):
   ```bash
   git clone https://github.com/sierra-research/tau2-bench
   cd tau2-bench
   pip install -e .
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

4. Your trajectory files will be saved in `data/tau2/simulations/` within the tau2-bench directory.

#### Cost Tracking (Optional but Recommended)
To enable fair comparisons between models with different pricing structures, we encourage submitting cost information:

1. **Calculate average cost per trajectory** for each domain by dividing total cost by number of trajectories run
2. **Include costs in USD** when creating your submission file using the optional `cost` field
3. **Document cost calculation method** in the `methodology.notes` field if using custom cost tracking

### Step 2: Document Any Framework Modifications or Task Omissions
If you made any changes to the tau2-bench framework or evaluation protocol, you **must** document these in your pull request:

#### Framework Modifications
If you modified the tau2-bench framework (user simulation, prompts, evaluation logic, etc.):
1. **Fork Documentation**: Provide a link to your tau2-bench fork if you made any code changes
2. **Change Summary**: Clearly describe what was modified and why in your pull request description
3. **Code References**: Link to specific commits, files, or line numbers where changes were made
4. **Reproducibility**: Ensure others can reproduce your results using your modified framework

#### Task Omissions
If you omitted any tasks from your evaluation runs:
1. **List Omitted Tasks**: Specify which tasks were skipped (by task ID or description)
2. **Reason for Omission**: Clearly explain why these tasks were omitted (e.g., technical limitations, resource constraints, model capabilities)
3. **Impact Assessment**: Describe how this might affect the interpretation of your results

**Important**: We strongly prefer submissions that link to tau2-bench forks rather than describing changes in text, as this ensures full transparency and reproducibility.

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
    └── my-awesome-model_telecom_default_gpt-4o_4trials.json
```

### Step 4: Add Your Trajectory Files
1. Copy your trajectory files from `data/tau2/simulations/` to your submission's `trajectories/` directory
2. Follow the naming convention: `{model-name}_{domain}_{mode}_{user-llm}_{num-trials}trials.json`
3. Add trajectory files for all domains you evaluated

### Step 5: Update the Manifest
Add your directory name to the `submissions` array in `public/submissions/manifest.json`:

```json
{
  "submissions": [
    "existing-submission-1_org_2024-12-01",
    "existing-submission-2_org_2024-12-15", 
    "my-awesome-model_mycompany_2025-01-15"
  ],
  "last_updated": "2025-01-15T12:00:00Z"
}
```

### Step 6: Submit Pull Request
1. Fork the [tau2-bench repository](https://github.com/sierra-research/tau2-bench)
2. Add your submission directory with submission.json, trajectory files, and update the manifest in the `web/leaderboard/` directory
3. Submit a pull request with:
   - Clear description of your model and results
   - **REQUIRED: Complete submission directory with trajectory files in `trajectories/` subdirectory**
   - Complete trajectory files for all evaluated domains (airline, retail, telecom)
   - **Documentation of any tau2-bench framework modifications or task omissions** (link to your fork, describe changes/omissions)
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
- [ ] Trajectory files follow naming convention
- [ ] Complete trajectory files for all evaluated domains
- [ ] **4 trials per domain** (check num-trials in trajectory files)
- [ ] Results verified against trajectory data
- [ ] Trajectory files contain valid tau2-bench output format
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
│       └── model1_telecom_default_gpt-4o_4trials.json
├── model2_org2_date/                   # Another submission
│   ├── submission.json
│   └── trajectories/
│       ├── model2_airline_default_claude-3-5-sonnet_4trials.json
│       └── ...
└── ...
```

## Example Submission

See the directory `public/submissions/EXAMPLE_new-model_example-org_2025-01-15/` for a complete example of what a submission should look like.

### Example Directory Structure
```
public/submissions/my-awesome-model_mycompany_2025-01-15/
├── submission.json
└── trajectories/
    ├── my-awesome-model_airline_default_gpt-4o_4trials.json
    ├── my-awesome-model_retail_default_gpt-4o_4trials.json
    └── my-awesome-model_telecom_default_gpt-4o_4trials.json
```

### Upload Instructions
1. Copy trajectory files from your tau2-bench `data/tau2/simulations/` directory
2. Rename them following the convention above if needed
3. Add them to your submission's `trajectories/` directory in your pull request
4. Do not compress or archive the files - upload the raw JSON files

### Generating Complete Trajectories
Refer to [Step 1: Evaluate Your Model and Generate Trajectories](#step-1-evaluate-your-model-and-generate-trajectories) for detailed instructions on running tau2-bench evaluations.

## Technical Details

The leaderboard component (`src/components/Leaderboard.jsx`) automatically:
1. Fetches the manifest file on page load
2. Loads each submission file listed in the manifest
3. Converts the JSON format to the internal data structure
4. Renders the leaderboard with all submissions
