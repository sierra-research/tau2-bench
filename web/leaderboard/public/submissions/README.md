# Tau2-Bench Leaderboard Submissions

This directory contains model evaluation results for the tau2-bench leaderboard. Each JSON file represents one model's performance across the benchmark domains.

## How to Submit Results

1. **Evaluate your model** using the [tau2-bench framework](https://github.com/sierra-research/tau2-bench)
2. **Create a JSON file** following the schema defined in `schema.json`
3. **Name your file** using the format: `{model_name}_{model_organization}_{submission_date}.json`
   - Example: `gpt-4.1_openai_2025-01-15.json`
   - Use lowercase, replace spaces with hyphens
4. **Update the manifest file** by adding your filename to `manifest.json`
5. **Submit a pull request** to this repository with both files added/modified
   - **Include trajectory links in your PR description** for verification (see below)
6. **Wait for review** - we'll validate your submission and merge if everything looks correct

## File Naming Convention

- Use lowercase letters, numbers, hyphens, and underscores only
- Format: `{model_name}_{model_organization}_{submission_date}.json`
- Examples:
  - `claude-3-7-sonnet_anthropic_2025-01-15.json`
  - `custom-model-v1_mycompany_2025-01-20.json`
  - `gpt-5_openai_2025-02-01.json`

## JSON Schema

Your submission must follow the schema defined in `schema.json`. Here's a quick example:

```json
{
  "model_name": "My-Amazing-Model-v1.0",
  "model_organization": "My Company",
  "submitting_organization": "My Company",
  "submission_date": "2025-01-15",
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
      "details": "Full evaluation with standard tau2-bench configuration"
    }
  }
}
```

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

## Important Notes

- **Pass@k scores**: Use `null` for missing data (e.g., if you only evaluated Pass@1)
- **Trajectories**: Set `trajectories_available: true` if you're including trajectory files in your submission, `false` otherwise
- **References**: Include links to papers, documentation, or other resources about your model (optional but recommended)
- **Cost information**: Optionally include the average cost in USD to run one trajectory in each domain using the `cost` field. This helps with fair comparisons between models with different pricing structures
- **Domains**: You don't need to evaluate all domains, but include at least one
- **Manifest file**: Don't forget to add your filename to `manifest.json` - the leaderboard loads files from this list
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

## Validation

Your JSON file will be automatically validated against the schema when you submit a pull request. Make sure:

- All required fields are present
- Email format is valid
- Scores are between 0-100 or `null`
- Cost values are positive numbers or `null` (if provided)
- Date format is YYYY-MM-DD
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
Your trajectory data should contain the raw evaluation output from tau2-bench, including:
- All conversation logs between your model and the user simulator
- Action sequences and tool calls made by your model
- Task completion status for each evaluation run
- Any relevant configuration or setup details

### Trajectory Naming Conventions
If you include trajectory files in your submission, they should follow these naming patterns:

**Standard Pattern**: `{model-identifier}_{domain}_{config}_{user-simulator}_{trials}.json`

**Examples of supported patterns**:
- `claude-3-7-sonnet-20250219_{domain}_default_gpt-4.1-2025-04-14_4trials.json`
- `gpt-4.1-2025-04-14_{domain}_default_gpt-4.1-2025-04-14_4trials.json`
- `gpt-4.1-mini-2025-04-14_{domain}_base_gpt-4.1-2025-04-14_4trials.json`
- `o4-mini-2025-04-16_{domain}_default_gpt-4.1-2025-04-14_4trials.json`
- `gpt-5_{domain}_default_gpt-4.1-2025-04-14_4trials.json`

Where `{domain}` will be replaced with: `airline`, `retail`, or `telecom`

**Note**: If your trajectory files use a different naming convention, please mention this in your pull request and we'll add support for your pattern.

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
1. Check the [tau2-bench documentation](https://github.com/sierra-research/tau2-bench)
2. Open an issue in this repository
3. Contact us at the email provided in the main README

Thank you for contributing to the tau2-bench leaderboard!
