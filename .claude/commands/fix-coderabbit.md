---
description: Analyze and implement CodeRabbit review fixes with context-aware evaluation and user approval.
---

## User Input

```text
$ARGUMENTS
```

You **MUST** process the CodeRabbit review comments provided above before proceeding.

## Goal

Systematically analyze CodeRabbit review comments, evaluate their validity against the project context and specifications, propose justified fixes, and implement approved changes. This command ensures review feedback is handled thoughtfully rather than blindly applied.

## Operating Constraints

- **Context-First**: Always understand the why before proposing fixes
- **Selective Application**: Not all review comments warrant changes—evaluate each critically
- **User Approval Required**: Never implement fixes without explicit user consent
- **Preserve Intent**: Fixes should align with original implementation goals and spec requirements

## Execution Steps

### Phase 1: Parse and Categorize Review Comments

1. **Extract review comments** from user input:
   - Parse each CodeRabbit comment
   - Identify the file path and line number(s) for each comment
   - Note the comment type (nitpick, minor issue, major issue, suggestion)
   - Comments will be separated by this string to delimit: "---"
   - Nitpick comments will not be delimited, but will start from "## Nitpicks"

2. **Create structured inventory**:
   ```
   | # | File | Lines | Type | Summary |
   |---|------|-------|------|---------|
   | 1 | src/file.py | 42-45 | Major | Missing error handling |
   | 2 | src/other.py | 10 | Nitpick | Naming convention |
   ```

3. **Categorize by severity**:
   - **Major Issues**: Security vulnerabilities, bugs, breaking changes, missing error handling
   - **Minor Issues**: Code quality, performance concerns, maintainability
   - **Nitpicks**: Style preferences, naming suggestions, comment improvements
   - **Suggestions**: Optional enhancements, alternative approaches

### Phase 2: Gather Context from Specifications

4. **Identify relevant specifications**:
   - Determine which spec folder(s) relate to the reviewed code
   - Look for spec references in file paths, imports, or comment context
   ```bash
   ls specs/
   ```

5. **Load specification context** (for each relevant spec):
   - Read `spec.md` for requirements and acceptance criteria
   - Read `plan.md` for architectural decisions and design rationale
   - Read `tasks.md` for implementation intent

   Focus on sections that inform the reviewed code:
   - Functional requirements that justify the implementation
   - Non-functional requirements (performance, security constraints)
   - Design decisions that explain current patterns
   - Edge cases that may justify seemingly unusual code

6. **Extract implementation rationale**:
   - Why was this approach chosen?
   - What constraints influenced the design?
   - Are there documented trade-offs?

### Phase 3: Analyze Changed Files

7. **Read each affected file**:
   - Load the full file context (not just reviewed lines)
   - Understand the surrounding code structure
   - Identify patterns and conventions in use

8. **For each review comment, evaluate**:

   **Validity Check**:
   - Is the observation technically correct?
   - Does it apply to this specific context?
   - Is there a false positive (e.g., intentional pattern, documented exception)?

   **Necessity Check**:
   - Does fixing this align with project goals?
   - Would the fix introduce regressions or conflicts?
   - Is this covered by existing tests or specs?
   - Does the spec explicitly require or forbid the current behavior?

   **Impact Assessment**:
   - What would change if we apply this fix?
   - Are there downstream effects?
   - Does it conflict with other requirements?

### Phase 4: Generate Fix Proposals

9. **For comments warranting fixes**, create proposals:

   ```markdown
   ## Proposed Fixes

   ### Fix 1: [Brief description]

   **Review Comment**: [Original CodeRabbit comment]
   **File**: `path/to/file.py:42-45`
   **Severity**: Major/Minor/Nitpick

   **Current Code**:
   ```python
   # existing code snippet
   ```

   **Proposed Change**:
   ```python
   # fixed code snippet
   ```

   **Reasoning**:
   - Why this fix is valid
   - How it aligns with spec requirements
   - What problem it solves

   **Spec Alignment**: References `specs/NNN-feature/spec.md` section X.Y
   ```

10. **For comments NOT being addressed**, document clearly:

    ```markdown
    ## Declined Fixes

    ### Declined 1: [Brief description]

    **Review Comment**: [Original CodeRabbit comment]
    **File**: `path/to/file.py:10`
    **Type**: Nitpick/Suggestion

    **Reason for Declining**:
    - [ ] False positive - code is intentionally written this way
    - [ ] Spec conflict - spec requires current behavior
    - [ ] Low value - change would add complexity without benefit
    - [ ] Out of scope - would require broader refactoring
    - [ ] Style preference - not aligned with project conventions

    **Detailed Explanation**: [Why this won't be changed]

    **Alternative Consideration**: [If applicable, what could be done instead]
    ```

### Phase 5: Present Summary and Request Approval

11. **Generate summary report**:

    ```markdown
    ## CodeRabbit Review Analysis Summary

    ### Statistics
    - Total comments analyzed: X
    - Proposed fixes: Y (Major: A, Minor: B, Nitpicks: C)
    - Declined fixes: Z

    ### By Category
    | Category | Proposed | Declined | Reason |
    |----------|----------|----------|--------|
    | Security | 2 | 0 | All valid |
    | Performance | 1 | 1 | One conflicts with spec |
    | Style | 0 | 3 | Project uses different convention |

    ### Spec References Used
    - specs/001-a2a-integration/spec.md
    - specs/002-evaluation-store/plan.md
    ```

12. **Request explicit approval**:

    Present the full proposal and ask:

    > **Ready to implement fixes?**
    >
    > Please review the proposed changes above. You can:
    > - **Approve all**: I'll implement all proposed fixes
    > - **Approve selected**: Specify which fix numbers to apply (e.g., "1, 3, 5")
    > - **Modify proposals**: Suggest changes to any fix before implementation
    > - **Decline all**: No changes will be made
    >
    > For declined fixes, let me know if you'd like me to reconsider any with additional context.

### Phase 6: Implement Approved Fixes

13. **After user approval**, implement changes:
    - Apply only the approved fixes
    - Use atomic edits (one logical change at a time)
    - Preserve existing code style and patterns
    - Add/update tests if the fix affects testable behavior

14. **Verify changes**:
    ```bash
    git diff
    ```
    - Confirm only intended changes were made
    - Run relevant tests if available
    - Check for syntax errors or regressions

15. **Report completion**:
    ```markdown
    ## Implementation Complete

    ### Applied Fixes
    - [x] Fix 1: Description
    - [x] Fix 3: Description
    - [x] Fix 5: Description

    ### Verification
    - Files modified: X
    - Tests affected: Y
    - Syntax check: Passed

    ### Next Steps
    - Run full test suite: `pytest`
    - Review changes: `git diff`
    - Commit when ready: `/commit`
    ```

## Evaluation Guidelines

### When to Accept a CodeRabbit Comment

- **Accept** if the issue is technically correct AND aligns with project specs
- **Accept** if it catches a genuine bug or security vulnerability
- **Accept** if it improves code quality without conflicting with requirements
- **Accept** if it fixes a deviation from documented conventions

### When to Decline a CodeRabbit Comment

- **Decline** if the current code is intentional and spec-justified
- **Decline** if fixing would break documented behavior
- **Decline** if it's purely stylistic and conflicts with project conventions
- **Decline** if the "improvement" adds complexity without clear benefit
- **Decline** if it's a false positive due to limited context understanding
- **Decline** if addressing it requires scope beyond the current PR

### Nitpick Handling Policy

Nitpicks deserve special consideration:
- Default to **declining** pure style nitpicks unless they violate explicit project standards
- **Accept** nitpicks that improve readability significantly
- **Accept** nitpicks that prevent future bugs or confusion
- Always explain why a nitpick was declined to help calibrate future reviews

## Important Notes

- **NEVER apply fixes blindly** - every change must be justified
- **ALWAYS read full file context** - line-level fixes without context cause bugs
- **RESPECT spec authority** - specifications override style preferences
- **PRESERVE test coverage** - don't break existing tests without reason
- **DOCUMENT declined fixes thoroughly** - transparency helps future reviews
- **ASK for clarification** if review comments are ambiguous
