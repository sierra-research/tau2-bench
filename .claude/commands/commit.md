---
description: Create git commits following repository conventions and historical commit patterns.
---

The user input to you can be provided directly by the agent or as a command argument - you **MUST** consider it before proceeding with the prompt (if not empty).

User input:

$ARGUMENTS

Given the optional user guidance in arguments, do this:

## Phase 1: Analyze Commit History and Patterns

1. **Extract commit patterns** from recent repository history:
   ```bash
   git log --pretty=format:"%s" -n 50
   ```

2. **Identify commit conventions** used in this repository:
   - Conventional Commits format? (feat:, fix:, chore:, docs:, test:, refactor:)
   - Scopes in parentheses? (feat(module): description)
   - Capitalization patterns
   - Punctuation style (period at end, no period, etc.)
   - Emoji usage patterns
   - Typical message length and structure
   - Multi-line message patterns (check with `git log --pretty=format:"%B" -n 10`)

3. **Build pattern template** based on observed conventions:
   - Most common prefixes and their meanings
   - Typical scope structure
   - Message tone (imperative, past tense, etc.)
   - Examples of well-formed messages from history

## Phase 2: Analyze Current Changes

4. **Review current repository state**:
   ```bash
   git status
   git diff --cached    # Staged changes
   git diff             # Unstaged changes
   ```

5. **Categorize changes** by logical grouping:
   - Group by feature/concern/module
   - Group by change type (new files, modifications, deletions)
   - Identify dependencies between changes (what must go together)
   - Flag any files that should not be committed (.env, secrets, large binaries)

6. **Map changes to commit types**:
   - New features → feat/feature commits
   - Bug fixes → fix commits
   - Documentation → docs commits
   - Tests → test commits
   - Refactoring → refactor commits
   - Chores (deps, config) → chore commits
   - Breaking changes → special handling

## Phase 3: Generate Commit Plan

7. **Create commit grouping strategy**:
   - Propose logical commits (1-N based on changes)
   - Each commit should:
     * Have a single, clear purpose
     * Be atomic (work independently)
     * Follow repository conventions
     * Include related changes only
   - Order commits by dependency (foundational first)

8. **Draft commit messages** for each proposed commit:
   - Match historical pattern identified in Phase 1
   - Follow conventional commits if detected
   - Use appropriate type/scope based on changes
   - Write clear, concise descriptions (brief and to the point - no unnecessary adjectives or filler words)
   - Keep subject line focused on WHAT changed, not WHY or HOW (save details for body)
   - **Body text style (if needed)**:
     * First paragraph: Single factual sentence describing the change - NO explanations of benefits, simplifications, or justifications
     * Bullet points: ONLY the most essential changes (4-5 maximum) - avoid exhaustive lists
     * NO closing paragraphs explaining future implications, benefits, or rationale
     * Focus on WHAT changed, not WHY it's better or HOW it will be used
   - Add breaking change notices if needed

9. **Present plan to user** with explicit file-to-commit mapping:
   ```
   === Commit Plan ===

   Commit 1: <type>(<scope>): <description>
   ┌─ Files ────────────────────────────────────────────
   │  M  src/module/file1.py
   │  M  src/module/file2.py
   │  A  src/module/file3.py
   └────────────────────────────────────────────────────
   Reasoning: <why these go together>

   Commit 2: <type>(<scope>): <description>
   ┌─ Files ────────────────────────────────────────────
   │  M  tests/test_module.py
   │  D  tests/old_test.py
   └────────────────────────────────────────────────────
   Reasoning: <why these go together>

   === File Summary ===
   Total files: 5
   ├─ Commit 1: 3 files (src/module/*)
   └─ Commit 2: 2 files (tests/*)
   ```

   **File status indicators**: M = Modified, A = Added, D = Deleted, R = Renamed

## Phase 4: Execute Commits (with user approval)

10. **Create commits sequentially**:
    - Stage files for each commit group
    - Create commit with drafted message
    - Verify commit created successfully
    - Continue to next commit

11. **Commit command format**:
    ```bash
    git add <files for this commit>
    git commit -m "<primary message>" -m "<body if needed>"
    ```

12. **Handle multi-line messages** using heredoc:
    ```bash
    git commit -m "$(cat <<'EOF'
    <type>(<scope>): <description>

    <body paragraph explaining what and why>

    <footer with breaking changes, issues, etc.>
    EOF
    )"
    ```

## Phase 5: Verification and Reporting

13. **Verify commits created**:
    ```bash
    git log --oneline -n <number of commits created>
    git status  # Should show clean or remaining unstaged files
    ```

14. **Report completion**:
    - List created commits with SHAs
    - Show any remaining unstaged changes
    - Confirm commits follow repository conventions
    - Suggest next steps (push, create PR, etc.)

## Special Cases

### If no historical patterns exist:
- Default to Conventional Commits specification
- Use imperative mood ("Add feature" not "Added feature")
- Keep first line under 72 characters
- Explain body in detail if needed

### If changes are too complex for automatic grouping:
- Present analysis to user
- Ask for guidance on grouping
- Iterate on commit plan before executing

### If conflicts or issues arise:
- Report the issue clearly
- Suggest remediation steps
- Do not force commit problematic changes

## User Guidance Integration

If user provided arguments (guidance):
- Respect any specific commit message preferences
- Honor requested grouping strategies
- Apply any custom scopes or types mentioned
- Override automatic detection when explicitly instructed

## Important Notes

- **IMPORTANT: NEVER add authoring information** to commit messages (no "Generated with Claude Code", "Co-Authored-By", or similar attributions) - these will be handled by the git commit workflow automatically
- **COMMIT MESSAGE STYLE**: Be terse and factual
  * Avoid explanatory phrases like "This simplifies...", "This provides...", "This prepares..."
  * No closing paragraphs about future benefits or implications
  * Limit bullet points to 4-5 most essential changes
  * State WHAT changed, not WHY it's better
- **NEVER commit** files with secrets, credentials, or sensitive data
- **ALWAYS verify** staged changes before committing
- **RESPECT** .gitignore patterns
- **FOLLOW** repository's commit conventions strictly
- **ASK** if uncertain about grouping or message style
- **CREATE** atomic commits that can be reverted independently