---
description: Create git commits for A2A work on spec branches and cherry-pick to feature/a2a-agent-integration branch.
---

The user input to you can be provided directly by the agent or as a command argument - you **MUST** consider it before proceeding with the prompt (if not empty).

User input:

$ARGUMENTS

Given the optional user guidance in arguments, do this:

## Overview: Cherry-Pick Workflow

This command implements the cherry-pick workflow for A2A development using spec branches:
1. Development happens on **spec branches** (e.g., `001-a2a-integration`, `002-evaluation-store`)
2. Commit changes on the current spec branch
3. Cherry-pick A2A-related commits to `feature/a2a-agent-integration` (clean branch for upstream PRs)
4. Spec branches are merged into `main` separately
5. The feature branch stays clean with only A2A-specific changes

## Phase 1: Verify Branch State

1. **Identify current branch**:
   ```bash
   git branch --show-current
   ```
   - Record the current branch name (expected to be a spec branch like `001-a2a-integration`)
   - If on `main` or `feature/a2a-agent-integration`, warn user that this workflow is designed for spec branches and ask how to proceed

2. **Verify feature branch exists**:
   ```bash
   git branch --list feature/a2a-agent-integration
   ```
   - If it doesn't exist, offer to create it from `upstream/main`

3. **Check for clean working tree on feature branch**:
   ```bash
   git stash list
   ```
   - Note any stashed changes that might conflict

## Phase 2: Analyze Commit History and Patterns

4. **Extract commit patterns** from recent repository history:
   ```bash
   git log --pretty=format:"%s" -n 50
   ```

5. **Identify commit conventions** used in this repository:
   - Conventional Commits format? (feat:, fix:, chore:, docs:, test:, refactor:)
   - Scopes in parentheses? (feat(module): description)
   - Capitalization patterns
   - Punctuation style (period at end, no period, etc.)
   - Emoji usage patterns
   - Typical message length and structure
   - Multi-line message patterns (check with `git log --pretty=format:"%B" -n 10`)

6. **Build pattern template** based on observed conventions:
   - Most common prefixes and their meanings
   - Typical scope structure
   - Message tone (imperative, past tense, etc.)
   - Examples of well-formed messages from history

## Phase 3: Analyze Current Changes

7. **Review current repository state**:
   ```bash
   git status
   git diff --cached    # Staged changes
   git diff             # Unstaged changes
   ```

8. **Categorize changes** by logical grouping:
   - Group by feature/concern/module
   - Group by change type (new files, modifications, deletions)
   - Identify dependencies between changes (what must go together)
   - Flag any files that should not be committed (.env, secrets, large binaries)

9. **Classify changes as A2A vs Non-A2A**:
   - **A2A changes** (will be cherry-picked): Code related to A2A protocol, agent integration, tau2-bench functionality
   - **Non-A2A changes** (stay on spec branch only): Tooling, dev workflow, local configs, speckit files, spec-specific changes
   - Ask user to confirm classification if uncertain

10. **Map changes to commit types**:
    - New features → feat(a2a) commits
    - Bug fixes → fix(a2a) commits
    - Documentation → docs(a2a) commits
    - Tests → test(a2a) commits
    - Refactoring → refactor(a2a) commits
    - Chores (deps, config) → chore(a2a) commits
    - Breaking changes → special handling

## Phase 4: Generate Commit Plan

11. **Create commit grouping strategy**:
    - Propose logical commits (1-N based on changes)
    - **CRITICAL**: Keep A2A changes in separate commits from non-A2A changes
    - Each commit should:
      * Have a single, clear purpose
      * Be atomic (work independently)
      * Follow repository conventions
      * Include related changes only
    - Order commits by dependency (foundational first)

12. **Draft commit messages** for each proposed commit:
    - **A2A commits MUST use (a2a) scope**: `feat(a2a): add protocol client`
    - Match historical pattern identified in Phase 2
    - Follow conventional commits if detected
    - Write clear, concise descriptions (brief and to the point)
    - Keep subject line focused on WHAT changed
    - **Body text style (if needed)**:
      * First paragraph: Single factual sentence describing the change
      * Bullet points: ONLY the most essential changes (4-5 maximum)
      * NO closing paragraphs explaining future implications
      * Focus on WHAT changed, not WHY it's better

13. **Present plan to user** with explicit file-to-commit mapping:
    ```
    === Commit Plan ===

    [A2A - WILL BE CHERRY-PICKED TO feature/a2a-agent-integration]
    Commit 1: feat(a2a): <description>
    ┌─ Files ────────────────────────────────────────────
    │  M  tau2_agent/agent.py
    │  M  tau2_agent/tools/run_tau2_evaluation.py
    │  A  tau2_agent/protocol.py
    └────────────────────────────────────────────────────
    Reasoning: <why these go together>

    [A2A - WILL BE CHERRY-PICKED TO feature/a2a-agent-integration]
    Commit 2: fix(a2a): <description>
    ┌─ Files ────────────────────────────────────────────
    │  M  tau2_agent/utils.py
    │  M  tests/test_tau2_agent/test_utils.py
    └────────────────────────────────────────────────────
    Reasoning: <why these go together>

    [NON-A2A - STAYS ON SPEC BRANCH ONLY]
    Commit 3: chore: <description>
    ┌─ Files ────────────────────────────────────────────
    │  M  specs/008-gcp-integration/spec.md
    │  A  .claude/commands/new-command.md
    └────────────────────────────────────────────────────
    Reasoning: <tooling/workflow change>

    === File Summary ===
    Total files: 7
    ├─ A2A commits (cherry-picked): 5 files
    │  ├─ Commit 1: 3 files (tau2_agent/*)
    │  └─ Commit 2: 2 files (tau2_agent/*, tests/*)
    └─ Non-A2A commits (spec branch only): 2 files
       └─ Commit 3: 2 files (specs/*, .claude/*)
    ```

    **File status indicators**: M = Modified, A = Added, D = Deleted, R = Renamed

## Phase 5: Execute Commits on Spec Branch (with user approval)

14. **Create commits sequentially on the current spec branch**:
    - Stage files for each commit group
    - Create commit with drafted message
    - Verify commit created successfully
    - **Record SHA of each A2A commit** for cherry-picking
    - Continue to next commit

15. **Commit command format**:
    ```bash
    git add <files for this commit>
    git commit -m "<primary message>" -m "<body if needed>"
    ```

16. **Handle multi-line messages** using heredoc:
    ```bash
    git commit -m "$(cat <<'EOF'
    <type>(a2a): <description>

    <body paragraph explaining what and why>

    <footer with breaking changes, issues, etc.>
    EOF
    )"
    ```

17. **Capture commit SHAs**:
    ```bash
    git log --format="%H" -n 1  # Get SHA of just-created commit
    ```

## Phase 6: Cherry-Pick to Feature Branch

18. **Switch to feature branch**:
    ```bash
    git checkout feature/a2a-agent-integration
    ```

19. **Verify feature branch is up to date** (optional but recommended):
    ```bash
    git fetch upstream
    git log --oneline HEAD..upstream/main  # Check if behind
    ```
    - If behind, ask user if they want to rebase first

20. **Cherry-pick each A2A commit**:
    ```bash
    git cherry-pick <sha1>
    git cherry-pick <sha2>
    # Or for a range:
    git cherry-pick <first-sha>^..<last-sha>
    ```

21. **Handle cherry-pick conflicts** if they occur:
    - Show conflict details
    - Offer options:
      * Resolve manually
      * Skip this commit
      * Abort cherry-pick
    - Guide user through resolution

22. **Return to spec branch**:
    ```bash
    git checkout <spec-branch-name>  # Return to the original spec branch recorded in Phase 1
    ```

## Phase 7: Verification and Reporting

23. **Verify commits on both branches**:
    ```bash
    # Commits on spec branch
    git log --oneline <spec-branch-name> -n <number of commits created>

    # Commits on feature branch
    git log --oneline feature/a2a-agent-integration -n <number of a2a commits>

    # Verify feature branch content relative to upstream
    git log feature/a2a-agent-integration --not upstream/main --oneline
    ```

24. **Report completion**:
    ```
    === Commit Summary ===

    On <spec-branch-name>:
    - <sha1> feat(a2a): description [CHERRY-PICKED]
    - <sha2> fix(a2a): description [CHERRY-PICKED]
    - <sha3> chore: tooling change [SPEC BRANCH ONLY]

    On feature/a2a-agent-integration:
    - <sha1'> feat(a2a): description
    - <sha2'> fix(a2a): description

    Feature branch is ready for PR to upstream.
    Spec branch will be merged to main via normal PR process.
    Currently on branch: <spec-branch-name>
    ```

25. **Suggest next steps**:
    - Push spec branch: `git push origin <spec-branch-name>`
    - Push feature: `git push origin feature/a2a-agent-integration`
    - Create/update PR for spec branch → main if needed
    - Create/update PR for feature branch → upstream if needed

## Special Cases

### If no A2A changes detected:
- Inform user no cherry-pick is needed
- Offer to run standard commit workflow instead

### If only A2A changes detected:
- All commits will be cherry-picked
- Proceed with standard workflow

### If feature branch doesn't exist:
```bash
git fetch upstream
git checkout -b feature/a2a-agent-integration upstream/main
```

### If cherry-pick results in empty commit:
- This means the change already exists on feature branch
- Skip with `git cherry-pick --skip`
- Report which commits were skipped and why

### If user wants to abort mid-workflow:
- Ensure we return to the original spec branch
- Clean up any partial cherry-pick state
- Report what was completed vs. skipped

## User Guidance Integration

If user provided arguments (guidance):
- Respect any specific commit message preferences
- Honor requested grouping strategies
- Apply any custom scopes or types mentioned
- Override automatic A2A classification when explicitly instructed
- Skip cherry-pick if user specifies `--no-cherry-pick`

## Important Notes

- **IMPORTANT: NEVER add authoring information** to commit messages (no "Generated with Claude Code", "Co-Authored-By", or similar attributions)
- **A2A COMMITS MUST use (a2a) scope** for easy identification in history
- **COMMIT MESSAGE STYLE**: Be terse and factual
  * Avoid explanatory phrases like "This simplifies...", "This provides..."
  * No closing paragraphs about future benefits
  * Limit bullet points to 4-5 most essential changes
  * State WHAT changed, not WHY it's better
- **NEVER commit** files with secrets, credentials, or sensitive data
- **ALWAYS verify** staged changes before committing
- **RESPECT** .gitignore patterns
- **FOLLOW** repository's commit conventions strictly
- **ASK** if uncertain about A2A vs non-A2A classification
- **CREATE** atomic commits that can be reverted independently
- **KEEP** the feature branch clean - only A2A-related changes
