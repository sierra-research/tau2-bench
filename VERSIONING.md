# Versioning Strategy & Release Process

This document outlines the versioning strategy and release process for τ²-bench.

## Semantic Versioning

We follow [Semantic Versioning (SemVer)](https://semver.org/) with the format `MAJOR.MINOR.PATCH`:

- **MAJOR**: Incompatible API changes, breaking changes to domain policies, or fundamental architecture changes
- **MINOR**: New features, new domains, backwards-compatible functionality additions
- **PATCH**: Bug fixes, documentation updates, performance improvements

### Examples

- `1.0.0` → `1.0.1`: Bug fix in evaluation metrics
- `1.0.0` → `1.1.0`: New domain added (e.g., healthcare)
- `1.0.0` → `2.0.0`: Breaking API change in agent interface

## Release Categories

### Major Releases (x.0.0)
- Significant architectural changes
- Breaking changes to APIs
- Major new capabilities
- Released when breaking changes accumulate (typically annually)

### Minor Releases (x.y.0)
- New domains or evaluation scenarios
- New CLI commands or major features
- Backwards-compatible API extensions
- Released as needed (every few months based on feature readiness)

### Patch Releases (x.y.z)
- Bug fixes and security patches
- Documentation improvements
- Performance optimizations
- Released as needed (typically within days of fixes)

## Release Process

We use **automated releases** via Release Please for consistency and efficiency, with manual releases as a fallback option.

### Automated Release Process (Recommended)

Our GitHub Actions workflow automatically handles releases when conventional commits are pushed to main:

1. **Development with Conventional Commits**
   ```bash
   git commit -m "feat(domains): add healthcare domain"
   git commit -m "fix(cli): resolve Unicode handling"
   git commit -m "feat!: redesign agent API (BREAKING CHANGE)"
   ```

2. **Automatic Release PR Creation**
   - Release Please analyzes commits since last release
   - Creates/updates a release PR with:
     - Updated `CHANGELOG.md`
     - Version bump in `pyproject.toml`
     - Generated release notes

3. **Review and Merge**
   - Review the automated release PR
   - Merge when ready to release
   - GitHub release is automatically created

4. **Optional: Manual Release Notes Enhancement**
   - Update `RELEASE_NOTES.md` with user-friendly content
   - Add migration guides for breaking changes

### Manual Release Process (Fallback)

For urgent releases or when automation isn't available:

1. **Pre-Release Testing**
   ```bash
   make test  # Run full test suite
   tau2 run --domain mock --num-tasks 1  # Quick integration test
   ```

2. **Update Files**
   - Update `version` in `pyproject.toml`
   - Add entry to `CHANGELOG.md`
   - Update `RELEASE_NOTES.md`

3. **Create Release**
   ```bash
   git add .
   git commit -m "chore: prepare release v1.1.0"
   git tag -a v1.1.0 -m "Release version 1.1.0"
   git push origin main
   git push origin v1.1.0
   ```

4. **Create GitHub Release**
   - Use content from `RELEASE_NOTES.md`
   - Attach any relevant artifacts

### Post-Release Activities

1. **Automated Actions** (handled by workflow)
   - GitHub release creation
   - Package building (ready for PyPI)
   - Tag creation

2. **Manual Follow-up**
   - Update `RELEASE_NOTES.md` with user-friendly content
   - Update leaderboard at tau-bench.com if needed
   - Social media announcements for major releases
   - Blog posts for significant features

3. **Optional: PyPI Publishing**
   ```bash
   # Uncomment PyPI section in .github/workflows/release.yml
   # Add PYPI_API_TOKEN to GitHub secrets
   ```

## Changelog Maintenance

### Automated Changelog Generation

Our Release Please workflow automatically generates `CHANGELOG.md` entries from conventional commits:

| Commit Type | Changelog Section | Example |
|-------------|-------------------|---------|
| `feat:` | Added | `feat(domains): add healthcare domain` |
| `fix:` | Fixed | `fix(cli): resolve Unicode handling` |
| `perf:` | Performance | `perf: optimize concurrent execution` |
| `docs:` | Documentation | `docs: update installation guide` |

### Manual Changelog Categories

When manually updating `CHANGELOG.md`, use these standardized categories:

- **Added**: New features, domains, or capabilities
- **Changed**: Changes in existing functionality  
- **Deprecated**: Soon-to-be removed features
- **Removed**: Features removed in this version
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes
- **Performance**: Performance improvements

### Entry Format

```markdown
## [1.1.0] - 2025-02-15

### Added
- New healthcare domain with 50 evaluation tasks
- Support for streaming responses in CLI
- Agent performance visualization dashboard

### Changed
- Improved error messages in submission validation
- Updated default LLM timeout from 30s to 60s

### Fixed
- Fixed memory leak in concurrent evaluations
- Resolved issue with Unicode characters in task descriptions
```

## Development Versions

For ongoing development, we use `-dev` suffix in `pyproject.toml`:

- **Development**: `0.2.1-dev` - Active development after v0.2.0 release
- **Pre-release**: `0.3.0-alpha.1` - Early testing (manual release)
- **Release Candidate**: `0.3.0-rc.1` - Final testing before release

### Current Practice
- Update to `x.y.z-dev` immediately after releasing `x.y.z`
- Use conventional commits during development
- Let Release Please handle version bumping for releases

## Backporting Policy

### Long-Term Support (LTS)

- Major versions (x.0.0) receive security updates for 1 year
- Critical bug fixes backported to last 2 minor versions
- Security patches backported to all supported versions

### Backport Criteria

- **Security vulnerabilities**: Always backported
- **Critical bugs**: Backported to supported versions
- **Data corruption issues**: Immediate backport
- **Performance regressions**: Case-by-case basis

## Deprecation Policy

### Timeline

1. **Announcement**: Feature marked as deprecated
2. **Warning Period**: 2 minor versions with warnings
3. **Removal**: Remove in next major version

### Communication

- Add deprecation warnings to code
- Document in CHANGELOG.md
- Include migration guide in RELEASE_NOTES.md
- Announce in GitHub discussions

## Automation Tools

### Recommended Tools

1. **Conventional Commits**: Standardize commit messages
2. **Release Please**: Automate changelog generation
3. **Semantic Release**: Automatic version bumping
4. **GitHub Actions**: Automate testing and releases

### Current GitHub Action Setup

Our actual workflow (`.github/workflows/release.yml`):

```yaml
name: Release
on:
  push:
    branches: [main]
permissions:
  contents: write
  pull-requests: write
jobs:
  release-please:
    runs-on: ubuntu-latest
    steps:
      - uses: google-github-actions/release-please-action@v4
        with:
          release-type: python
          package-name: tau2
          version-file: pyproject.toml
          include-v-in-tag: true
          changelog-types: |
            [
              {"type":"feat","section":"Added","hidden":false},
              {"type":"fix","section":"Fixed","hidden":false},
              {"type":"perf","section":"Performance","hidden":false},
              {"type":"docs","section":"Documentation","hidden":false}
            ]
```

This workflow:
- ✅ Automatically creates release PRs
- ✅ Generates changelog from conventional commits  
- ✅ Handles version bumping in `pyproject.toml`
- ✅ Creates GitHub releases
- 🔲 PyPI publishing (ready but commented out)

## Emergency Releases

For critical security or data corruption issues:

1. **Immediate Response**: Fix on main branch
2. **Fast Track**: Skip normal review process
3. **Hotfix Release**: Increment patch version
4. **Communication**: Immediate notification to users
5. **Post-Mortem**: Document incident and prevention

---

This versioning strategy ensures predictable, reliable releases while maintaining backwards compatibility and clear communication with users.
