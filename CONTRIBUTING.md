# Contributing to τ-Bench

Thank you for your interest in contributing to τ-bench! This document provides guidelines to help you make clean, reviewable contributions that can be easily integrated into the project.

## 🚀 Quick Start

1. **Open an Issue First** (recommended): Before starting work, open an issue to discuss your proposed changes
2. **Fork & Clone**: Fork the repository and clone it locally
3. **Follow Branch Conventions**: Use descriptive branch names following our naming conventions
4. **Make Clean Commits**: Write clear commit messages and keep changes focused
5. **Test Your Changes**: Ensure all tests pass and add new tests as needed
6. **Open a Clean PR**: Follow our PR template and guidelines

## 📝 Types of Contributions

### Core Framework Contributions
- **Agent implementations**: New agent types or improvements to existing agents
- **Environment enhancements**: Tools, evaluation metrics, or orchestration improvements
- **Performance optimizations**: Caching, parallel processing, or efficiency improvements
- **Bug fixes**: Resolving issues in the core framework

### Domain Contributions
- **New domains**: Complete domain implementations with tasks, tools, and policies
- **Domain improvements**: Enhanced tools, tasks, or policy refinements for existing domains
- **Domain-specific agents**: Specialized agents optimized for particular domains

### Experimental Contributions
- **Research code**: Novel approaches, prototypes, and experimental features
- **Location**: All experimental code goes in `src/experiments/`
- **Requirements**: Each experiment needs its own README with clear documentation
- **Status**: Experimental code is provided as-is and may not be fully supported

### Documentation & Infrastructure
- **Documentation improvements**: README updates, API docs, tutorials
- **Testing enhancements**: New tests, test infrastructure improvements
- **CI/CD improvements**: Workflow enhancements, automation improvements

## 🎯 Before You Start: Open an Issue

**We strongly recommend opening an issue before starting work**, especially for:
- New features or significant changes
- New domain implementations
- Large refactoring efforts
- Experimental contributions

### Issue Template
When opening an issue, please include:
- **Problem/Goal**: What problem are you solving or what feature are you adding?
- **Proposed Solution**: High-level approach you plan to take
- **Impact**: What components will be affected?
- **Timeline**: Expected development timeline
- **Dependencies**: Any external dependencies or blockers

This helps us:
- Provide early feedback and guidance
- Avoid duplicate work
- Ensure alignment with project goals
- Suggest the best approach for your contribution

## 🌿 Branch Naming Conventions

Use clear, descriptive branch names following these patterns:

### Core Framework Changes
- `feature/description` - New features
- `fix/issue-description` - Bug fixes
- `refactor/component-name` - Code refactoring
- `perf/optimization-description` - Performance improvements

### Domain-Specific Changes
- `domain/domain-name/feature-description` - New domain or domain features
- `domain/domain-name/fix-description` - Domain bug fixes

### Experimental Contributions
- `experiment/experiment-name` - New experimental features
- `experiment/experiment-name/enhancement` - Improvements to existing experiments

### Documentation & Infrastructure
- `docs/description` - Documentation updates
- `test/description` - Test improvements
- `ci/description` - CI/CD improvements

### Examples
```bash
# Good branch names
feature/agent-memory-system
fix/environment-tool-timeout
domain/healthcare/patient-lookup-tools
experiment/multi-agent-collaboration
docs/contributing-guide-update
test/domain-integration-tests

# Avoid
my-changes
fix
update
new-stuff
```

## 🔧 Development Setup

### 1. Environment Setup
```bash
# Clone your fork
git clone https://github.com/your-username/tau2-bench.git
cd tau2-bench

# Install with dev tools (pytest, ruff, pre-commit — required for committing)
uv sync --extra dev

# Verify installation
uv run tau2 check-data
```

Install additional extras as needed for the area you're working on:

```bash
uv sync --extra voice          # voice/audio-native features
uv sync --extra knowledge      # banking_knowledge domain (retrieval pipeline)
uv sync --extra gym            # gymnasium RL interface
uv sync --extra experiments    # plotting libs for src/experiments/
uv sync --all-extras           # everything
```

This requires [uv](https://docs.astral.sh/uv/getting-started/installation/). The Python version (3.12) is pinned via `.python-version` — uv will download it automatically if needed.

### 2. Development Dependencies
The project uses several tools for code quality (included in the `dev` extra):
- **uv**: Package and project management
- **Ruff**: Linting and code formatting
- **pytest**: Testing framework
- **pre-commit**: Git hook management

### 3. Environment Variables
Copy `.env.example` to `.env` and configure your API keys for testing.

## 🧪 Testing Requirements

### Running Tests
```bash
# Run all tests
make test

# Run specific test categories
pytest tests/test_domains/  # Domain tests
pytest tests/test_agent.py  # Agent tests
pytest tests/test_environment.py  # Environment tests
```

### Test Requirements for PRs
- **Existing tests must pass**: All current tests should continue to pass
- **New functionality needs tests**: Add tests for new features or bug fixes
- **Domain contributions**: Include comprehensive domain-specific tests
- **Experimental code**: Basic smoke tests recommended but not required

### Test Coverage Guidelines
- **Core framework changes**: Aim for good test coverage of new functionality
- **Domain implementations**: Test all tools, tasks, and policy interactions
- **Bug fixes**: Include regression tests to prevent the bug from reoccurring

## 📋 Code Quality Standards

### Code Formatting and Linting
We use **Ruff** for both linting and formatting:

```bash
# Check linting
make lint

# Format code
make format

# Auto-fix linting issues
make lint-fix

# Run both linting and formatting
make check-all
```

### Code Style Guidelines
- **Line length**: 88 characters (configured in pyproject.toml)
- **Import organization**: Use Ruff's import sorting
- **Type hints**: Encouraged for new code, especially in core framework
- **Docstrings**: Required for public APIs and complex functions

### Commit Message Guidelines
Write clear, concise commit messages:
```bash
# Good commit messages
feat: add memory system to agent base class
fix: resolve environment tool timeout issues
docs: update domain contribution guidelines
test: add integration tests for retail domain

# Avoid
fixed stuff
updates
wip
```

## 🔍 Pull Request Guidelines

### Before Opening a PR
- [ ] All tests pass locally (`make test`)
- [ ] Code follows style guidelines (`make check-all` passes)
- [ ] New functionality is tested
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive

### PR Title and Description
**Title Format**: `type: brief description`

**Description Template**:
```markdown
## Summary
Brief description of the changes made.

## Changes Made
- List of specific changes
- Include any breaking changes
- Note any new dependencies

## Testing
- How you tested the changes
- What test cases were added/modified
- Any manual testing performed

## Documentation
- Any documentation updates made
- Links to relevant docs or issues

## Checklist
- [ ] Tests pass (`make test`)
- [ ] Code follows style guidelines (`make check-all`)
- [ ] Documentation updated
- [ ] Breaking changes noted
```

### PR Review Process
1. **Automated checks**: CI tests and code quality checks must pass
2. **Maintainer review**: Code review focusing on:
   - Correctness and functionality
   - Code quality and maintainability
   - Test coverage and documentation
   - Alignment with project goals
3. **Feedback incorporation**: Address review feedback promptly
4. **Final approval**: Maintainer approval required for merge

## 🎯 Specific Contribution Guidelines

### Domain Contributions
When contributing a new domain:
- **Complete implementation**: Include all required components (tools, tasks, policy, tests)
- **Documentation**: Comprehensive README with domain overview, API docs, and examples
- **Test coverage**: Full test suite covering all domain functionality
- **Data validation**: Ensure all domain data is properly validated

### Experimental Contributions
For `src/experiments/` contributions:
- **Self-contained**: Keep experimental code isolated within the experiments directory
- **Documentation**: Include detailed README explaining the experiment and usage
- **Dependencies**: Manage dependencies carefully to avoid conflicts with core framework
- **Status clarity**: Clearly mark experimental status and limitations

### Agent Contributions

There are two types of agent contributions:

#### Core Agents (`src/tau2/agent/`)
Core agents are part of the official tau2 framework and are maintained by Sierra. Core agent contributions:
- Require thorough review and approval
- Must implement `HalfDuplexAgent` or `FullDuplexAgent`
- Are registered in `src/tau2/registry.py`
- See `src/tau2/agent/README.md` for the full developer guide


## 🤝 Getting Help

- **GitHub Issues**: For bugs, feature requests, and general questions
- **Documentation**: Check existing docs and README files first

## 📜 Code of Conduct

- Be respectful and constructive in all interactions
- Focus on the technical aspects of contributions
- Help maintain a welcoming environment for all contributors
- Follow generally accepted open source collaboration practices

## 🎉 Recognition

Contributors who make significant contributions may be:
- Added to the project's contributor list
- Mentioned in release notes

Thank you for contributing to τ-bench! Your efforts help advance the field of conversational AI evaluation.
