# Release Notes

Welcome to the τ²-bench release notes! Here you'll find user-friendly summaries of what's new, what's changed, and what you need to know for each release.

## Version 0.2.0 - Web-Based Leaderboard 🌐

**Release Date**: October 6, 2025

### 🌟 Major New Feature: Live Leaderboard

We're excited to announce the biggest addition to τ²-bench since launch - a comprehensive web-based leaderboard system that's now live!

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
The leaderboard at **tau-bench.com** makes τ²-bench results accessible to:
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
tau2 submit prepare data/tau2/simulations/your_results*.json --output ./my_submission

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

This release focuses on making τ²-bench easier to install and configure for everyone.

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

### What is τ²-bench?

τ²-bench is a comprehensive framework for evaluating conversational agents in realistic, dual-control environments. This groundbreaking release provides everything you need to benchmark your AI agents across multiple customer service domains.

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

*Ready to benchmark your conversational agents? Visit [tau-bench.com](https://tau-bench.com) to see the leaderboard and get started with τ²-bench today!*