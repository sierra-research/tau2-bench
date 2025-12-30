---
name: documentation
description: |
  Documentation skill for docstrings, comments, and code documentation.

  TRIGGER KEYWORDS: docstring, docstrings, comment, comments, documentation,
  documenting, document, docs, README, inline comment, code comment, JSDoc,
  type annotation, API docs, module docs, describe function, explain code,
  add documentation, update documentation, fix documentation, missing docs,
  Google style, documentation style, doc format.

  Use this skill when:
  - Writing, adding, updating, or fixing docstrings
  - Adding or modifying comments (inline, block, or section comments)
  - Documenting functions, classes, methods, or modules
  - Reviewing or improving documentation quality
  - Refactoring code that needs documentation updates
  - Creating or updating README files
  - Adding type annotations with descriptions

  Enforces Google-style docstrings and meaningful, non-redundant comments.
---

# Documentation Skill

## Purpose

Ensure docstrings and comments are clear, concise, and provide real value. Documentation describes **current behavior** without referencing changes, fixes, or previous implementations.

---

## Core Principles

### 1. Docstrings Describe the Contract

Docstrings explain **what** a function does and **how to use it**, not how it works internally.

> "A docstring should give enough information to write a call to the function without reading the function's code." — Google Python Style Guide

### 2. Comments Explain the Why

Comments are for non-obvious logic, business rules, or tricky implementation details.

> "Never describe the code. Assume the person reading the code knows Python better than you do." — Google Python Style Guide

### 3. Document Current State Only

Never reference:
- What the code "used to do"
- That something was "fixed" or "updated"
- The change being made in the current commit

Documentation should read as if it was always this way.

---

## Docstring Requirements

### When Docstrings Are Required

A docstring is **mandatory** for:
- Public API functions and methods
- Non-trivial functions (>10 lines or complex logic)
- Functions with non-obvious behavior
- Classes (describe what an instance represents)
- Modules (describe purpose and key exports)

A docstring is **optional** for:
- Simple, obvious helper functions
- Private methods with clear names
- Properties with self-explanatory names

### Google-Style Format

Use this format for all docstrings. See [docstring-examples.md](docstring-examples.md) for complete examples.

```python
def function_name(param1: str, param2: int = 0) -> bool:
    """One-line summary ending with a period.

    Extended description if needed. Explain the purpose and any
    important behavioral details the caller should know.

    Args:
        param1: Description of the first parameter.
        param2: Description with default noted if non-obvious.

    Returns:
        Description of return value semantics.

    Raises:
        ValueError: When param1 is empty.
        TypeError: When param2 is not an integer.
    """
```

### Section Order

When present, sections must appear in this order:
1. Summary line (required)
2. Extended description (optional)
3. Examples (optional)
4. Args (if parameters exist)
5. Returns or Yields (if applicable)
6. Raises (if exceptions are raised)

### Formatting Rules

| Element | Rule |
|---------|------|
| Summary | One line, ≤80 chars, ends with period |
| Args | `param_name: Description.` — no type if annotated |
| Multi-line | Hanging indent of 4 spaces |
| Blank lines | One blank line between sections |
| Self/cls | Never document in Args section |

---

## Comment Requirements

### When Comments Are Valuable

Add comments for:
- **Business logic**: Why this calculation exists
- **Non-obvious constraints**: Why this limit is 100
- **Workarounds**: Why we do X instead of Y (with context)
- **Algorithm notes**: Key insight for complex logic
- **Section breaks**: Separating logical phases in long functions

### When Comments Are Noise

Do NOT add comments for:
- Restating what code does (`# increment counter`)
- Describing obvious operations (`# return the result`)
- Narrating the change (`# fixed bug where...`)
- TODO items without context (`# TODO: fix this`)

### Comment Formatting

```python
# Single-line comments: 2+ spaces before #, 1 space after
result = calculate_value()  # Explain non-obvious detail

# Block comments precede the code they describe
# This algorithm uses X approach because Y constraint
# requires minimizing memory usage over speed.
for item in large_collection:
    process(item)
```

---

## Anti-Patterns

See [anti-patterns.md](anti-patterns.md) for detailed examples. Key violations:

### Never Reference Changes

```python
# BAD: References the change
"""Process data using the new streaming approach."""
"""Now handles edge case where input is empty."""
# Fixed: previously crashed on None input

# GOOD: Describes current behavior
"""Process data using streaming to minimize memory usage."""
"""Handle empty input by returning an empty result."""
```

### Never Narrate Code

```python
# BAD: Restates the code
x = x + 1  # Add one to x
return result  # Return the result

# GOOD: Explains why (if needed at all)
x = x + 1  # Offset for 1-based indexing expected by API
```

### Never Over-Document

```python
# BAD: Obvious from signature
def get_user_by_id(user_id: int) -> User:
    """Get a user by their ID.

    Args:
        user_id: The ID of the user.

    Returns:
        The user.
    """

# GOOD: Adds value
def get_user_by_id(user_id: int) -> User:
    """Retrieve a user from the database by ID.

    Raises:
        UserNotFoundError: If no user exists with the given ID.
    """
```

---

## Updating Existing Documentation

When modifying code with existing docstrings or comments:

1. **If behavior changed**: Rewrite the docstring to describe new behavior
2. **If behavior unchanged**: Leave documentation as-is
3. **If documentation was wrong**: Fix it to match actual behavior
4. **Never add**: "Updated to...", "Now does...", "Fixed..."

The documentation should read as authoritative truth about current behavior, with no archaeological record of changes.

---

## Quick Reference

### Docstring Template

```python
"""Summary line under 80 characters ending with period.

Extended description if the summary isn't sufficient.

Args:
    param1: Description of first parameter.
    param2: Description of second parameter.

Returns:
    Description of what is returned.

Raises:
    ExceptionType: When this exception occurs.
"""
```

### Comment Checklist

Before adding a comment, ask:
- [ ] Does this explain WHY, not WHAT?
- [ ] Would a reader need this to understand the code?
- [ ] Is this information not obvious from the code itself?
- [ ] Does this avoid referencing changes or fixes?

If any answer is "no", reconsider the comment.
