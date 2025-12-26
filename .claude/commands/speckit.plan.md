---
description: Execute the implementation planning workflow using the plan template to generate design artifacts.
handoffs: 
  - label: Create Tasks
    agent: speckit.tasks
    prompt: Break the plan into tasks
    send: true
  - label: Create Checklist
    agent: speckit.checklist
    prompt: Create a checklist for the following domain...
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

1. **Setup**: Run `.specify/scripts/bash/setup-plan.sh --json` from repo root and parse JSON for FEATURE_SPEC, IMPL_PLAN, SPECS_DIR, BRANCH. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **Load context**: Read FEATURE_SPEC and `.specify/memory/constitution.md`. Load IMPL_PLAN template (already copied).

   **Checklist gate**: Check if `FEATURE_DIR/checklists/requirements.md` exists.
   - If missing: Warn user - "No requirements checklist found. Consider running `/speckit.checklist requirements` first to validate spec quality."
   - If exists but incomplete: Warn user - "Requirements checklist has incomplete items. Review before proceeding."
   - Continue regardless (soft gate), but document warning in output.

3. **Execute plan workflow**: Follow the structure in IMPL_PLAN template to:
   - Fill Technical Context (mark unknowns as "NEEDS CLARIFICATION")
   - Fill Constitution Check section from constitution
   - Evaluate gates (ERROR if violations unjustified)
   - Phase 0: Generate research.md (resolve all NEEDS CLARIFICATION)
   - Phase 1: Generate data-model.md, contracts/, quickstart.md
   - Phase 1: Update agent context by running the agent script
   - Re-evaluate Constitution Check post-design

4. **Stop and report**: Command ends after Phase 2 planning. Report branch, IMPL_PLAN path, and generated artifacts.

## Phases

### Phase 0: Outline & Research

1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:

   ```text
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using the structured template from `.specify/templates/research-template.md`:

   **CRITICAL**: Use the exact template format to enable automated parsing by `/speckit.tasks` and `/speckit.verify`.

   **Required sections**:
   - `## Dependencies`: Runtime and Development dependency tables with version constraints
   - `## Decision Registry`: Each decision as `### DEC-XXX:` block with:
     - **Decision**: Concrete choice made
     - **Pattern**: Code pattern to verify (searchable string)
     - **Verify In**: File path or glob where pattern should appear
     - **Rationale**: Why this choice
     - **Alternatives Rejected**: What wasn't chosen and why
     - **Verification Points**: Checkable items (become checklist items)

   **Version Management**:
   - Pin all dependencies with version constraints (>=X.Y.Z or ==X.Y.Z)
   - Link to package registry for verification (PyPI, npm, etc.)
   - Document version constraints and incompatibilities

**Output**: research.md with all NEEDS CLARIFICATION resolved, structured for automated verification

### Phase 1: Design & Contracts

**Prerequisites:** `research.md` complete

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Agent context update**:
   - Run `.specify/scripts/bash/update-agent-context.sh claude`
   - These scripts detect which AI agent is in use
   - Update the appropriate agent-specific context file
   - Add only new technology from current plan
   - Preserve manual additions between markers

**Output**: data-model.md, /contracts/*, quickstart.md, agent-specific file

## Key rules

- Use absolute paths
- ERROR on gate failures or unresolved clarifications
