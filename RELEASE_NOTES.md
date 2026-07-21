# Release Notes

Welcome to the τ-bench release notes! Here you'll find user-friendly summaries of what's new, what's changed, and what you need to know for each release.

## Version 1.0.1 - Banking Grading Fixes ⚖️

**Release Date**: July 2026

### ⚠️ banking_knowledge scores change with this release

1.0.1 bundles the recent `banking_knowledge` grading and task fixes. Together they correct
a systematic penalty that zeroed rewards for correct-but-cautious agent behavior, plus
task-data issues surfaced by recent runs.
**Scores computed before and after this release are not comparable on banking_knowledge.**
All other domains are unaffected.

#### What was wrong

1. **Extra read calls zeroed the reward** (#329). Every `call_discoverable_agent_tool` call
   was logged to a DB table that participates in the reward's DB-hash comparison. An agent
   that did one prudent verification read not present in the golden trajectory — e.g.
   listing a user's accounts after opening one, or checking a replacement-card order it just
   placed — failed the task, even though the knowledge base encourages such reads. Reads are
   now logged only when the task's golden trajectory requires them, so "agent must verify X"
   assertions still discriminate while extra validation reads no longer poison the hash.
2. **Int-vs-float argument spelling changed DB hashes** (#397). `25` and `25.0` are the same
   JSON number but produced different deterministic record IDs and DB states. Numeric
   arguments are now normalized.
3. **Tasks 077–086 gold trajectories were not agent-realizable** (#402). The lost/stolen
   card scenarios omitted reads any real agent must perform; the golden trajectories now
   include them.
4. **Bank account transactions were returned oldest-first, contradicting the docs** (#403).
   The knowledge base documents `get_bank_account_transactions_9173` as most-recent-first,
   but the tool returned raw DB insertion order, making the "dispute the earliest duplicate"
   tie-breaker unresolvable from tool output on the duplicate-charge tasks (083–085). Output
   is now sorted by date descending (stable sort), and task_085's fixture rows were aligned
   to the same tie-break convention as 083/084. No gold actions changed.
5. **Contradictory cash-back rate in the Platinum Rewards knowledge doc** (#388). The
   document stated two different rates for the same card; the numbers now agree.
6. **task_074 under-refunded Light Blue ATM fees against its own policy docs** (#374). The
   Light Blue Account docs grant two free out-of-network and two free foreign ATM
   withdrawals per month, but the gold refund honored only one of each ($8.00 instead of
   $14.50). The fixture's fee descriptions also embedded the answer key — annotations like
   "(SHOULD BE FREE - 1ST OF 2)" and "(AFTER 2 FREE)" told the agent which fees were
   erroneous. The gold credit is now the policy-faithful $14.50 and all twelve annotated
   descriptions are scrubbed to plain statement text, so the agent must derive the free
   allowance from the policy docs. Four "AFTER FREE ALLOWANCE" annotations of the same
   kind on the task_072/073 Light Green accounts are scrubbed too (description-only, no
   grading impact). Unlike the grading fixes, this changes a gold
   value: a trajectory that reproduced the old $8.00 refund fails task_074 under 1.0.1,
   while policy-faithful $14.50 refunds now pass.

#### Measured impact

We re-graded the leaderboard `banking_knowledge` trajectory sets under the new scheme.
Under the grading-scheme fixes (items 1–5) every score change is upward — no
previously-passing simulation fails — and the shift is model-dependent, so relative
rankings change. The task_074 gold correction (item 6) can move individual task_074 trials
in either direction; in the trajectory sets inspected, no trial matched the old $8.00 gold
(so nothing flips down) and one gemini-3-1-pro-preview trial that applied the correct
$14.50 refund flips to passing:

| Model | pass^1 (old) | pass^1 (new) | Δ pass^1 | pass^4 (old) | pass^4 (new) | flips |
|---|---|---|---|---|---|---|
| gpt-5-5 | 37.37 | 46.39 | +9.02 | 20.62 | 27.84 | +35/-0 |
| gpt-5-4 | 30.67 | 39.43 | +8.76 | 16.49 | 21.65 | +34/-0 |
| gpt-5-2 | 24.74 | 32.22 | +7.48 | 11.34 | 18.56 | +29/-0 |
| claude-opus-4-7 | 25.26 | 30.15 | +4.89 | 12.37 | 15.46 | +19/-0 |
| claude-opus-4-6 | 24.48 | 27.32 | +2.84 | 10.31 | 11.34 | +11/-0 |
| gemini-3-flash | 20.62 | 27.32 | +6.7 | 4.12 | 7.22 | +26/-0 |
| gemini-3-1-pro-preview | 22.54 | 26.03 | +3.49 | 8.42 | 9.28 | +14/-0 |
| claude-sonnet-4-5 | 22.42 | 25.26 | +2.84 | 10.31 | 10.31 | +11/-0 |
| claude-opus-4-5 | 21.39 | 24.74 | +3.35 | 8.25 | 11.34 | +13/-0 |
| gemini-3-pro | 15.72 | 18.04 | +2.32 | 4.12 | 4.12 | +9/-0 |
| grok-4-2 | 17.57 | 18.04 | +0.47 | 7.29 | 8.25 | +2/-0 |
| grok-4-fast | 14.18 | 15.72 | +1.54 | 4.12 | 4.12 | +6/-0 |
| gemini-2-5-pro | 12.76 | 13.66 | +0.9 | 1.08 | 1.03 | +4/-0 |
| grok-4-1-fast | 12.4 | 13.14 | +0.74 | 5.21 | 5.15 | +3/-0 |
| gpt-5-2-none | 11.08 | 12.63 | +1.55 | 4.12 | 4.12 | +1/-0 |
| glm-5-think | 9.79 | 9.79 | 0 | 3.09 | — | +0/-0 |
| qwen3.5-397b-a17b-think | 9.79 | 9.79 | 0 | 5.15 | 5.15 | +0/-0 |

_No simulation flipped from pass to fail; every reward change is upward. pass^k values are recomputed counting infrastructure-error simulations as failed trials (the leaderboard convention); the sub-0.1-point pass^4 dips for gemini-2-5-pro and grok-4-1-fast come from that convention alignment, not from any score flip. glm-5-think's trajectory file contains only 3 trials for some tasks, so pass^4 is not recomputable; its previous value is retained. NL-assertion judgments are preserved from the original grading runs (the v1.0.1 fixes are all environment-side), so the re-grade is fully deterministic._

Attribution across all flipped simulations: essentially every flip comes from the
extra-read logging fix (#329); two flips are additionally covered by the task 077–086 gold
trajectory fix (#402, either fix alone suffices); the numeric normalization (#397) changed
no final leaderboard scores on its own but removes a latent source of spurious failures.

#### Re-grading your own results

```bash
tau2 evaluate-trajs --fresh-tasks path/to/results.json -o regraded/
```

`--fresh-tasks` (new in 1.0.1) grades against the current task definitions instead of the
ones embedded in your results file. Re-grading also now applies the same per-task read-log
allowlist as live grading and tolerates cosmetic tool-output drift in trajectories recorded
before 1.0.1.

#### Reproducing pre-1.0.1 behavior

Most of the fixes above were merged to `main` on July 14–15, 2026 — ahead of this release
and without a version bump. That means "1.0.0" installed from `main` after that window
already behaves like 1.0.1 on `banking_knowledge`. If you need the pre-fix behavior (for
example, to reproduce `banking_knowledge` scores published before this release), install
from the `pre-v1.0.1` tag — the last commit before the fix series landed
(`b51a6d69e26f0e94a9173e2e80fe8735a8dff650`):

```bash
pip install git+https://github.com/sierra-research/tau2-bench@pre-v1.0.1
```

The `v1.0.0` tag predates this commit by several months; `pre-v1.0.1` is the recommended pin
because it includes everything shipped since then except the grading changes.

Any earlier 2026 commit of `main` grades `banking_knowledge` the same way. For example, from
commit `2be6916` (April 2026) through `pre-v1.0.1` the evaluator, environment, orchestrator,
banking tools, and database are byte-for-byte identical; changes in that window are
simulation-side only (the task_053 `user_tools` fix from #328, the default banking retrieval
config changing from `bm25` to `alltools`, and retrieval prompt updates). Any pin in that
range reproduces the same pre-1.0.1 scores when re-grading recorded trajectories.

---

## Version 0.2.1 - Reinforcement Learning Support 🤖

**Release Date**: November 2025

### 🎮 Gymnasium Integration

τ-bench now supports reinforcement learning with a standard Gymnasium-compatible interface!

#### 🌟 What's New
- **Train RL Agents**: Use `AgentGymEnv` and `UserGymEnv` with popular RL frameworks
- **Interactive Play Mode**: New `tau2 play` command lets you control the agent or user manually
- **Train/Test Splits**: Standardized task splits across all domains for proper evaluation
- **Backward Compatible**: Use `base` task split to evaluate on the complete original τ-bench task set
- **Enforce Communication Protocol**: Optionally enforce communication protocol rules (e.g., no mixed messages with text and tool calls)

#### 🚀 Getting Started
```bash
# Try interactive play mode
tau2 play

# Use programmatically with gym interface
from tau2.gym import AgentGymEnv, UserGymEnv
```

See the [Gym Documentation](src/tau2/gym/README.md) for detailed usage examples and API reference.

---

## Version 0.2.0 - Web-Based Leaderboard 🌐

**Release Date**: October 6, 2025

### 🌟 Major New Feature: Live Leaderboard

We're excited to announce the biggest addition to τ-bench since launch - a comprehensive web-based leaderboard system that's now live!

#### 🚀 What's New
- **Interactive Leaderboard**: Browse and compare model performance across all domains
- **Live at tau-bench.com**: Fully deployed and accessible to the community
- **Submission Management**: Easy submission validation and verification process
- **Trajectory Visualization**: Explore conversation flows and agent decisions
- **Mobile Support**: Full responsive design for viewing on any device
- **Automated Deployment**: GitHub Pages integration with CI/CD pipeline
- **Professional Branding**: Logo assets for all major LLM providers

#### 🔧 For Researchers & Developers
- Submit your results directly through the web interface
- Visual comparison of model performance metrics across domains
- Export functionality for research papers and presentations
- Direct links to submission data and trajectories
- Real-time leaderboard updates with new submissions

#### 🌍 Community Impact
The leaderboard at **tau-bench.com** makes τ-bench results accessible to:
- Researchers comparing agent performance
- Industry practitioners evaluating models
- Academic institutions teaching agent evaluation
- Open source community building better agents

### 🛠️ Technical Improvements
- **Enhanced Infrastructure**: Robust deployment pipeline
- **Better Asset Management**: Optimized image loading and branding
- **Mobile Optimization**: Responsive design across all devices
- **Improved Validation**: More comprehensive submission checking

### 🚀 Getting Started with the Leaderboard

1. **Visit**: [tau-bench.com](https://tau-bench.com)
2. **Explore**: Browse current model rankings and performance
3. **Submit**: Follow the submission guide to add your model
4. **Compare**: Analyze how your agent performs against others

### 📊 Submission Process

Ready to showcase your agent? Our submission system makes it easy:

```bash
# Run complete evaluation on all domains
tau2 run --domain retail --agent-llm your-model --user-llm gpt-4 --num-trials 4
tau2 run --domain airline --agent-llm your-model --user-llm gpt-4 --num-trials 4  
tau2 run --domain telecom --agent-llm your-model --user-llm gpt-4 --num-trials 4

# Prepare submission
tau2 submit prepare data/simulations/your_results*.json --output ./my_submission

# Validate before submitting
tau2 submit validate ./my_submission
```

### ⚡ Performance & Reliability
- **Fast Loading**: Optimized for quick access to results
- **Mobile-First**: Designed for accessibility on any device
- **Always Available**: Robust hosting ensures consistent uptime
- **Regular Updates**: Automatic deployment of new features

### 📈 What's Next

With the leaderboard now live, we're focusing on:
- Enhanced trajectory analysis tools
- More sophisticated evaluation metrics
- Additional domain support
- Community-driven features and improvements

---

## Version 0.1.3 - Stability & Performance 🔧

**Release Date**: August 26, 2025

### 🐛 Key Fixes

#### LLM Integration Improvements
- **Fixed LLM argument parsing**: Resolved issues with complex LLM configurations
- **Removed problematic assertions**: Eliminated default natural language assertion checks that were causing evaluation failures

#### Impact
These fixes significantly improve the reliability of evaluations, especially when using advanced LLM configurations or custom parameters.

### 🚀 Upgrade Notes
- Simply update to v0.1.3 - no breaking changes
- Existing evaluation configs will work without modification
- Performance should be more consistent across different LLM providers

---

## Version 0.1.2 - Installation & Usability 📦

**Release Date**: July 17, 2025

### 🌟 Installation Made Easy

This release focuses on making τ-bench easier to install and configure for everyone.

#### New Installation Features
- **Default editable install**: `pip install -e .` is now the recommended method
- **Flexible data directory**: Set `TAU2_DATA_DIR` for custom installations
- **Smart fallbacks**: Automatic detection of data directory location
- **Installation verification**: New `tau2 check-data` command

#### Enhanced CLI Experience
```bash
# Verify your installation
tau2 check-data

# Control task count more precisely
tau2 run --domain airline --num-tasks 10 --agent-llm gpt-4
```

#### Developer Experience
- **Better task management**: Improved task name display and filtering
- **Clearer error messages**: More helpful feedback when things go wrong
- **Simplified setup**: Fewer configuration steps required

### 🚀 Migration Guide
If you have an existing installation:
1. Reinstall with `pip install -e .`
2. Run `tau2 check-data` to verify setup
3. Remove any manual data directory configurations (now automatic)

---

## Version 0.1.1 - Quick Fix 🔧

**Release Date**: June 12, 2025

### 🐛 Domain Viewer Fix

Fixed critical issues with the domain documentation viewer:
- `tau2 domain <domain>` now works correctly
- Resolved CLI command execution problems
- Improved error handling for domain-specific operations

---

## Version 0.1.0 - Initial Public Release 🚀

**Release Date**: June 12, 2025

### What is τ-bench?

τ-bench is a comprehensive framework for evaluating conversational agents in realistic, dual-control environments. This groundbreaking release provides everything you need to benchmark your AI agents across multiple customer service domains.

### 🌟 Core Features

#### Multi-Domain Evaluation
- **4 realistic domains**: Mock, Airline, Retail, and Telecom
- Each domain includes realistic policies, tools, and evaluation tasks
- Industry-standard scenarios for comprehensive agent testing

#### Easy-to-Use Command Line Interface
```bash
# Run your first evaluation in minutes
tau2 run --domain airline --agent-llm gpt-4 --user-llm gpt-4 --num-trials 1 --num-tasks 5
```

#### Dual-Control Environment
- **Realistic interactions**: Both agent and user can interact with the system
- **AI-powered user simulation**: Creates authentic conversation scenarios
- **Comprehensive metrics**: Pass@k success rates and detailed performance analysis

### 🔧 For Developers

#### Agent Development Made Simple
- Clean API for implementing custom agents
- Comprehensive documentation for each domain
- Interactive domain viewer at `http://127.0.0.1:8004/redoc`
- Example implementations included

#### Flexible & Extensible
- Support for any LLM provider via LiteLLM
- Configurable concurrency and trial settings
- Redis-based caching for cost optimization
- Extensible domain system for custom scenarios

### 🔬 Research Applications

#### Advanced Evaluation Features
- **Ablation studies**: No-user mode and oracle-plan mode
- **Policy format comparison**: Standard vs workflow policies
- **Comprehensive logging**: Every interaction captured for analysis
- **Statistical rigor**: Multi-trial evaluation with proper metrics

### 🚀 Getting Started

1. **Install**: `pip install -e .`
2. **Configure**: Set up your LLM API keys in `.env`
3. **Run**: `tau2 run --domain mock --agent-llm gpt-4 --user-llm gpt-4 --num-trials 1`
4. **Explore**: `tau2 view` to see your results

### 🤝 Community & Research

- **Paper**: [Read our research paper](https://arxiv.org/abs/2506.07982)
- **Blog**: [Learn more about the methodology](https://sierra.ai/blog/benchmarking-agents-in-collaborative-real-world-scenarios)
- **Open Source**: Full source code available on GitHub
- **Active Development**: Regular updates and community contributions

### ⚠️ Requirements

- **Python 3.10+**: Modern Python version required
- **LLM API Access**: OpenAI, Anthropic, or other LiteLLM-supported providers
- **Optional**: Redis for LLM call caching (disabled by default)

---

*Ready to benchmark your conversational agents? Visit [tau-bench.com](https://tau-bench.com) to see the leaderboard and get started with τ-bench today!*