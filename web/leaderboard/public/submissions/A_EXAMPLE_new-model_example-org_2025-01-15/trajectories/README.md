# Example Trajectory Files

This directory would contain trajectory files for the Example-Model-v2.0 submission.

Expected files (keep original filenames from tau2-bench):
- `example-model_airline_default_gpt-4o_4trials.json`
- `example-model_retail_default_gpt-4o_4trials.json`
- `example-model_telecom_default_gpt-4o_4trials.json`
- `example-model_banking_knowledge_gpt-4o_4trials.json`

Then add a `trajectory_files` mapping in `submission.json` to tell the leaderboard which file is which domain.

Note: Actual trajectory files are not included in this example as they would be large JSON files containing the full evaluation traces.
