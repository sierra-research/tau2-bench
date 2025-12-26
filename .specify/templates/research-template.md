# Research: [FEATURE]

**Feature Branch**: `[###-feature-name]`
**Date**: [DATE]
**Spec**: [link to spec.md]

<!--
  This template is filled in by the /speckit.plan command during Phase 0 research.

  IMPORTANT: This document has a structured format that enables:
  1. Auto-generation of research-compliance.md checklist by /speckit.tasks
  2. Post-implementation verification by /speckit.verify
  3. Dependency version tracking and validation

  Follow the exact format for Decision and Dependency blocks to enable parsing.
-->

## Dependencies

<!--
  CRITICAL: All external dependencies MUST be listed here with pinned versions.

  Format: Each dependency block follows this structure for automated parsing.
  The speckit.tasks command will extract these to verify against package manifests.
  The speckit.verify command will validate installed versions match.
-->

### Runtime Dependencies

| Package | Version | Purpose | Verified |
|---------|---------|---------|----------|
| [package-name] | [>=X.Y.Z or ==X.Y.Z] | [Why needed] | [ ] |

<!--
  VERSION PINNING GUIDELINES:
  - Use ==X.Y.Z for critical dependencies where exact version matters
  - Use >=X.Y.Z for dependencies where minimum version is required
  - Use >=X.Y.Z,<A.B.0 for dependencies with known breaking changes in future major versions
  - Document WHY if using anything other than latest stable

  VERIFICATION:
  - [ ] = Not yet verified compatible
  - [x] = Tested and verified compatible
  - Link to PyPI/npm/etc. for version confirmation
-->

**Example:**
```markdown
| ddtrace | >=4.0.0 | APM and LLM Observability | [x] [PyPI](https://pypi.org/project/ddtrace/) |
| pydantic | >=2.0.0,<3.0.0 | Data validation | [x] |
```

### Development Dependencies

| Package | Version | Purpose | Verified |
|---------|---------|---------|----------|
| [package-name] | [>=X.Y.Z] | [Why needed] | [ ] |

### Version Constraints

<!--
  Document any version constraints or incompatibilities discovered during research.
  This section is critical for preventing dependency conflicts.
-->

| Constraint | Reason | Impact |
|------------|--------|--------|
| [e.g., ddtrace >=4.0.0 requires Python 3.9+] | [Why this matters] | [What to do about it] |

## Decision Registry

<!--
  CRITICAL: Each decision block MUST follow this exact format for automated parsing.

  The speckit.tasks command will:
  1. Extract all DEC-XXX blocks
  2. Generate research-compliance.md checklist items
  3. Link tasks to decisions via [Research §DEC-XXX] tags

  The speckit.verify command will:
  1. Search codebase for PATTERN strings
  2. Report violations where pattern is not found
  3. Validate VERIFY_FILES contain expected patterns
-->

### DEC-001: [Decision Title]

**Decision**: [Concrete choice - what we decided to use/do]

**Pattern**:
```
[Code pattern, import statement, or configuration that MUST appear in implementation]
[This is what speckit.verify will search for]
```

**Verify In**: `[file path or glob pattern where this should appear, e.g., src/**/*.py]`

**Rationale**: [Why this choice was made]

**Alternatives Rejected**:
- [Alternative 1]: [Why rejected]
- [Alternative 2]: [Why rejected]

**Verification Points**:
- [ ] [Specific check 1 - becomes checklist item]
- [ ] [Specific check 2 - becomes checklist item]

---

### DEC-002: [Next Decision Title]

**Decision**: [Concrete choice]

**Pattern**:
```
[Code pattern to verify]
```

**Verify In**: `[file path or glob]`

**Rationale**: [Why]

**Alternatives Rejected**:
- [Alternative]: [Why rejected]

**Verification Points**:
- [ ] [Specific verification check]

---

## Integration Notes

<!--
  Document how dependencies and decisions interact.
  This helps identify potential conflicts or integration requirements.
-->

### Dependency Interactions

| Dependency A | Dependency B | Interaction | Notes |
|--------------|--------------|-------------|-------|
| [e.g., ddtrace] | [e.g., litellm] | [e.g., Auto-instrumentation] | [e.g., Use patch(litellm=True)] |

### Decision Dependencies

| Decision | Depends On | Reason |
|----------|------------|--------|
| [DEC-002] | [DEC-001] | [Why DEC-002 requires DEC-001 first] |

## Open Questions

<!--
  Questions that arose during research but aren't blocking.
  Move to Decision Registry once resolved.
-->

| Question | Status | Resolution |
|----------|--------|------------|
| [Question] | [OPEN/RESOLVED] | [Answer if resolved] |

## Research Complete

<!--
  Final checklist before proceeding to planning/implementation.
-->

- [ ] All NEEDS CLARIFICATION items resolved
- [ ] All dependencies verified compatible
- [ ] All decisions have verification points
- [ ] No open blocking questions
- [ ] Ready for /speckit.tasks to generate research-compliance.md
