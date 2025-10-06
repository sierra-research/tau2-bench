# Changelog & Release Automation Guide

This guide explains how to automate changelog generation and release management for τ²-bench using modern CI/CD tools.

## Overview

We've set up automated changelog generation using:
- **Conventional Commits**: Standardized commit message format
- **Release Please**: Google's automated release tool
- **GitHub Actions**: CI/CD automation
- **Semantic Versioning**: Predictable version numbering

## Conventional Commits

### Format

Use this standardized commit message format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description | Changelog Section | Version Bump |
|------|-------------|-------------------|--------------|
| `feat` | New feature | Added | Minor |
| `fix` | Bug fix | Fixed | Patch |
| `perf` | Performance improvement | Performance | Patch |
| `docs` | Documentation changes | Documentation | Patch |
| `refactor` | Code refactoring | Refactoring | Patch |
| `test` | Test additions/changes | Testing | None |
| `chore` | Maintenance tasks | Maintenance | None |
| `ci` | CI/CD changes | CI/CD | None |
| `build` | Build system changes | Build | None |
| `revert` | Revert previous commit | Reverted | Patch |

### Breaking Changes

For breaking changes, add `!` after the type or include `BREAKING CHANGE:` in the footer:

```bash
# Method 1: Exclamation mark
git commit -m "feat!: change agent API interface"

# Method 2: Footer
git commit -m "feat: add new domain support

BREAKING CHANGE: Agent interface now requires domain parameter"
```

### Examples

```bash
# New feature
git commit -m "feat(domains): add healthcare domain with 50 tasks"

# Bug fix
git commit -m "fix(cli): resolve Unicode handling in task descriptions"

# Performance improvement
git commit -m "perf(evaluation): optimize concurrent task execution"

# Documentation
git commit -m "docs: update installation instructions for Windows"

# Breaking change
git commit -m "feat!: redesign agent configuration API"
```

## Release Please Setup

### How It Works

1. **Commit Analysis**: Scans commits since last release
2. **Version Calculation**: Determines next version based on commit types
3. **Changelog Generation**: Creates/updates CHANGELOG.md
4. **Release PR**: Opens pull request with changes
5. **Release Creation**: Creates GitHub release when PR is merged

### Configuration

The workflow is configured in `.github/workflows/release.yml`:

```yaml
# Key configuration options
release-type: python          # Python project type
package-name: tau2           # Your package name
version-file: pyproject.toml # Version location
include-v-in-tag: true      # Creates v1.0.0 tags
```

### Customization

You can customize changelog sections by editing the `changelog-types` in the workflow file:

```yaml
changelog-types: |
  [
    {"type":"feat","section":"🚀 Features","hidden":false},
    {"type":"fix","section":"🐛 Bug Fixes","hidden":false},
    {"type":"perf","section":"⚡ Performance","hidden":false},
    {"type":"docs","section":"📚 Documentation","hidden":false}
  ]
```

## Workflow Process

### Day-to-Day Development

1. **Write Code**: Develop your features/fixes
2. **Commit with Convention**: Use conventional commit format
3. **Push to Branch**: Create feature branch and push
4. **Create PR**: Open pull request to main branch
5. **Review & Merge**: Code review and merge to main

### Automated Release Process

1. **Trigger**: Push to main branch triggers workflow
2. **Analysis**: Release Please analyzes commits
3. **Release PR**: Automatically creates/updates release PR
4. **Review**: Team reviews the proposed changes
5. **Merge**: Merging the release PR creates the release
6. **Publish**: Optionally publishes to PyPI automatically

### Manual Release (if needed)

```bash
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG.md manually
# 3. Commit changes
git commit -m "chore: prepare release v1.1.0"

# 4. Create and push tag
git tag -a v1.1.0 -m "Release version 1.1.0"
git push origin v1.1.0

# 5. Create GitHub release from tag
```

## Best Practices

### Commit Messages

✅ **Good Examples:**
```bash
feat(domains): add telecom workflow policy support
fix(cli): handle missing config file gracefully  
docs(readme): update installation instructions
perf(evaluation): reduce memory usage in concurrent runs
test(domains): add integration tests for airline domain
```

❌ **Bad Examples:**
```bash
update stuff              # Too vague
fixed bug                # No context
WIP                      # Work in progress
asdf                     # Meaningless
```

### Development Workflow

1. **Feature Branches**: Always work in feature branches
2. **Small Commits**: Make focused, atomic commits
3. **Clear Messages**: Write descriptive commit messages
4. **Test First**: Ensure tests pass before committing
5. **Review Process**: Use pull requests for all changes

### Release Management

1. **Regular Releases**: Aim for regular release cadence
2. **Review Release PRs**: Always review automatically generated release PRs
3. **Test Releases**: Test release candidates thoroughly
4. **Communicate Changes**: Announce significant releases
5. **Monitor Issues**: Watch for issues after releases

## Troubleshooting

### Common Issues

**Release Please not triggering:**
- Check that commits follow conventional format
- Ensure workflow has proper permissions
- Verify branch protection rules

**Version not bumping correctly:**
- Review commit types and their version impact
- Check for BREAKING CHANGE annotations
- Verify conventional commit format

**Changelog missing entries:**
- Ensure commits use recognized types
- Check that commits are reachable from main
- Verify changelog-types configuration

### Debugging

```bash
# Check recent commits format
git log --oneline -10

# Validate conventional commit format
npx commitizen init cz-conventional-changelog --save-dev --save-exact

# Test release please locally (requires npx)
npx release-please release-pr --repo-url=https://github.com/your-org/tau2-bench
```

## Advanced Configuration

### Custom Release Types

Create `.release-please-manifest.json`:

```json
{
  ".": "0.0.1"
}
```

Create `release-please-config.json`:

```json
{
  "release-type": "python",
  "packages": {
    ".": {
      "package-name": "tau2",
      "changelog-sections": [
        {"type": "feat", "section": "Features"},
        {"type": "fix", "section": "Bug Fixes"},
        {"type": "perf", "section": "Performance Improvements"}
      ]
    }
  }
}
```

### Monorepo Support

For projects with multiple packages:

```json
{
  "packages": {
    "packages/core": {"package-name": "tau2-core"},
    "packages/cli": {"package-name": "tau2-cli"},
    "packages/web": {"package-name": "tau2-web"}
  }
}
```

### Pre/Post Release Hooks

Add custom scripts to `pyproject.toml`:

```toml
[tool.release-please]
pre-release-script = "scripts/pre-release.sh"
post-release-script = "scripts/post-release.sh"
```

## Integration with PyPI

To automatically publish to PyPI on release:

1. **Create PyPI API Token**: Generate token in PyPI account settings
2. **Add GitHub Secret**: Store token as `PYPI_API_TOKEN` in repository secrets
3. **Uncomment Publishing Step**: Enable the PyPI publishing section in the workflow

```yaml
# Uncomment these lines in .github/workflows/release.yml
- name: Publish to PyPI
  if: ${{ steps.release.outputs.release_created }}
  env:
    TWINE_USERNAME: __token__
    TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
  run: twine upload dist/*
```

## Monitoring and Analytics

### Release Metrics

Track these metrics to improve your release process:

- **Release Frequency**: How often you release
- **Time to Release**: From commit to release
- **Hotfix Rate**: Percentage of patch releases
- **Rollback Rate**: Failed releases requiring rollback

### Tools for Monitoring

- **GitHub Insights**: Built-in repository analytics
- **Release Dashboard**: Custom dashboard for release metrics
- **Automated Notifications**: Slack/Discord integration for releases

---

With this automation setup, your changelog will be automatically maintained, versions will be semantically versioned, and releases will be streamlined and consistent!
