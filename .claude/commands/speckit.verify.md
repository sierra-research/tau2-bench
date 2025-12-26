---
description: Verify implementation compliance with researched patterns, libraries, and decisions from research.md.
handoffs:
  - label: Run Analysis
    agent: speckit.analyze
    prompt: Run consistency analysis across all artifacts
    send: true
  - label: View Checklists
    agent: speckit.implement
    prompt: Check current checklist status
    send: false
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Goal

Perform automated post-implementation verification that the codebase matches the patterns, libraries, and decisions documented in `research.md`. This command runs code-level checks that complement the manual `research-compliance.md` checklist.

## Operating Constraints

**STRICTLY READ-ONLY**: Do **not** modify any files. Output a structured verification report. Suggest fixes but do not apply them automatically.

**Verification Authority**: The `research.md` document is the source of truth for:
- Dependency versions (must match package manifests)
- Code patterns (must appear in specified locations)
- Technical decisions (must be implemented as documented)

## Execution Steps

### 1. Initialize Verification Context

Run `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` from repo root and parse JSON for FEATURE_DIR and AVAILABLE_DOCS.

Derive absolute paths:
- RESEARCH = FEATURE_DIR/research.md
- CHECKLIST = FEATURE_DIR/checklists/research-compliance.md

**Abort** if research.md does not exist (nothing to verify against).

For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

### 2. Detect Project Type & Package Manifest

Identify the project's package management system by checking for:

| File | Type | Version Check Command |
|------|------|----------------------|
| `pyproject.toml` | Python (uv/poetry) | Parse `[project.dependencies]` or `[tool.poetry.dependencies]` |
| `requirements.txt` | Python (pip) | Parse `package==version` lines |
| `package.json` | Node.js | Parse `dependencies` and `devDependencies` |
| `Cargo.toml` | Rust | Parse `[dependencies]` |
| `go.mod` | Go | Parse `require` block |
| `Gemfile` | Ruby | Parse `gem` lines |

Store detected manifest path(s) for dependency verification.

### 3. Parse research.md

Extract verification targets from research.md:

#### A. Dependencies

Parse tables under `## Dependencies` sections:
- `### Runtime Dependencies` table
- `### Development Dependencies` table

For each row, extract:
- Package name
- Version constraint (e.g., `>=4.0.0`, `==1.2.3`, `>=1.0.0,<2.0.0`)
- Purpose (for context in report)

#### B. Decisions

Parse decision blocks matching pattern `### DEC-\d{3}:`:

For each decision, extract:
- Decision ID (e.g., `DEC-001`)
- Title
- **Pattern** code block (the pattern to search for)
- **Verify In** path/glob (where to search)
- **Verification Points** checklist items

#### C. Version Constraints

Parse `### Version Constraints` table for known incompatibilities.

### 4. Dependency Verification

For each dependency in research.md:

1. **Check presence**: Is the package listed in the detected manifest?
2. **Check version**: Does the manifest version satisfy the research.md constraint?

**Version Comparison Logic**:
- `>=X.Y.Z`: Manifest version must be >= X.Y.Z
- `==X.Y.Z`: Manifest version must be exactly X.Y.Z
- `>=X.Y.Z,<A.B.0`: Manifest version must be in range

**Report format for each dependency**:
```
| Package | Research | Manifest | Status |
|---------|----------|----------|--------|
| ddtrace | >=4.0.0 | 4.1.0 | PASS |
| pydantic | >=2.0.0 | 1.10.0 | FAIL: Version mismatch |
| httpx | >=0.28.0 | (missing) | FAIL: Not found |
```

### 5. Pattern Verification

For each decision with a **Pattern** and **Verify In** field:

1. **Search**: Use grep/ripgrep to find the pattern in specified path(s)
2. **Report**: Found locations with file:line references

**Pattern Search Logic**:
- If pattern is a code block, search for key identifiable strings
- Handle multi-line patterns by searching for distinctive lines
- Support glob patterns in Verify In (e.g., `src/**/*.py`)

**Report format for each decision**:
```
### DEC-001: [Title]

**Pattern**: `ddtrace.patch(litellm=True)`
**Verify In**: `src/**/*.py`

**Status**: PASS
**Found in**:
- src/instrumentation/tracing.py:42
- src/main.py:15

---

### DEC-002: [Title]

**Pattern**: `LLMObs.enable(agentless_enabled=True)`
**Verify In**: `src/observability/*.py`

**Status**: FAIL
**Expected in**: src/observability/*.py
**Not found**: Pattern not detected in any matching files

**Suggestion**: Add LLMObs initialization in src/observability/llm_obs.py
```

### 6. Cross-Reference Checklist

If `research-compliance.md` exists, cross-reference:
- Compare automated findings with checklist items
- Identify checklist items that can be auto-verified vs. manual-only
- Report checklist completion status

### 7. Produce Verification Report

Output a Markdown report with the following structure:

```markdown
# Research Compliance Verification Report

**Feature**: [Feature name from FEATURE_DIR]
**Verified Against**: research.md
**Date**: [Current date]

## Summary

| Category | Total | Pass | Fail | Skip |
|----------|-------|------|------|------|
| Dependencies | X | X | X | 0 |
| Decisions | X | X | X | X |
| **Overall** | X | X | X | X |

**Compliance Score**: X% (Pass / (Pass + Fail))

## Dependency Verification

| Package | Research Version | Manifest Version | Status | Notes |
|---------|------------------|------------------|--------|-------|
| ... | ... | ... | PASS/FAIL | ... |

### Missing Dependencies
[List packages in research.md but not in manifest]

### Version Mismatches
[List packages with version conflicts]

## Decision Verification

### DEC-001: [Title]
- **Status**: PASS/FAIL
- **Pattern**: `[pattern searched]`
- **Verify In**: `[path/glob]`
- **Found**: [file:line locations or "Not found"]
- **Action Required**: [None / Specific fix needed]

### DEC-002: [Title]
...

## Verification Points Summary

| ID | Decision | Verification Point | Auto-Verifiable | Status |
|----|----------|-------------------|-----------------|--------|
| RC001 | DEC-001 | [point] | Yes/No | PASS/FAIL/MANUAL |

## Checklist Cross-Reference

**Checklist Path**: [path to research-compliance.md]
**Checklist Status**: X/Y items complete

| Checklist Item | Auto-Verified | Manual Required |
|----------------|---------------|-----------------|
| DEP001 | PASS | - |
| RC001 | PASS | - |
| RC002 | - | Requires manual review |

## Recommendations

### Critical (Must Fix Before Release)
1. [Critical issues]

### High (Should Fix)
1. [High priority issues]

### Low (Consider Fixing)
1. [Low priority suggestions]

## Next Actions

- [ ] Fix dependency version mismatches in [manifest]
- [ ] Add missing pattern for DEC-XXX in [file]
- [ ] Complete manual verification for checklist items: [list]
- [ ] Run `/speckit.verify` again after fixes
```

### 8. Exit Criteria

- **All PASS**: Report success, suggest proceeding to release/merge
- **Any FAIL**: Report failures, block recommendation, suggest fixes
- **Manual items pending**: List items requiring human verification

## Verification Patterns by Language

### Python

```bash
# Check pyproject.toml dependencies
grep -E "^[package-name]" pyproject.toml

# Check for import patterns
grep -rn "from ddtrace" src/
grep -rn "ddtrace.patch" src/

# Check version in installed packages
uv pip show [package] | grep Version
```

### Node.js

```bash
# Check package.json dependencies
jq '.dependencies["package-name"]' package.json

# Check for import/require patterns
grep -rn "require('package')" src/
grep -rn "from 'package'" src/
```

### General

```bash
# Pattern search with context
grep -rn --include="*.py" "pattern" src/

# Glob-based search
find src -name "*.py" -exec grep -l "pattern" {} \;
```

## Operating Principles

### Verification Guidelines

- **NEVER modify files** (this is read-only verification)
- **NEVER skip failures** (all mismatches must be reported)
- **Prefer precision over recall** (avoid false positives)
- **Provide actionable suggestions** (specific file:line fixes)
- **Support incremental verification** (can run repeatedly during development)

### Pattern Matching

- Use exact string matching for specific patterns
- Use regex for flexible patterns (document in research.md)
- Handle multi-line patterns by matching key lines
- Report partial matches as warnings, not failures

### Version Comparison

- Parse semantic versions correctly (1.10.0 > 1.9.0)
- Handle version ranges (>=, <, ==, ~=, ^)
- Report pre-release versions explicitly
- Warn on major version differences

## Context

$ARGUMENTS
