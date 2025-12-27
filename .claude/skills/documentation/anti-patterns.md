# Documentation Anti-Patterns

Examples of documentation violations to avoid.

---

## 1. Referencing Changes or Fixes

Documentation should describe current behavior as authoritative truth.

### Docstrings

```python
# BAD: References the change
def process_data(data: bytes) -> Result:
    """Process data using the new streaming approach.

    Updated to handle large files without loading into memory.
    Now supports files over 4GB.
    """

# BAD: References what was fixed
def validate_input(value: str) -> bool:
    """Validate input string.

    Fixed: previously returned True for empty strings.
    Now correctly rejects empty input.
    """

# GOOD: Describes current behavior only
def process_data(data: bytes) -> Result:
    """Process data using streaming to minimize memory usage.

    Handles arbitrarily large inputs by processing in chunks.
    """

# GOOD: States the behavior as fact
def validate_input(value: str) -> bool:
    """Validate input string.

    Rejects empty strings and strings containing only whitespace.
    """
```

### Comments

```python
# BAD: References the fix
x = max(0, value)  # Fixed: now prevents negative values

# BAD: References old behavior
timeout = 30  # Changed from 60 to reduce wait time

# BAD: References the PR/commit
result = process(data)  # Added in PR #123 to handle edge case

# GOOD: Explains the constraint
x = max(0, value)  # API requires non-negative values

# GOOD: Explains why this value
timeout = 30  # Balances responsiveness with server load

# GOOD: No comment needed if obvious, or explain the why
result = process(data)  # Required before validation step
```

---

## 2. Narrating Code

Comments should explain WHY, not WHAT. The code shows what.

### Obvious Operations

```python
# BAD: Restates the code
counter = 0  # Initialize counter to zero
counter += 1  # Increment counter
return result  # Return the result
items.append(item)  # Add item to list

# GOOD: No comment needed for obvious code
counter = 0
counter += 1
return result
items.append(item)

# GOOD: Comment only if there's a non-obvious reason
counter = 0  # Reset for each batch to track per-batch metrics
```

### Control Flow

```python
# BAD: Describes the obvious
if user is None:  # Check if user is None
    return False  # Return False if no user
else:  # Otherwise
    return True  # Return True

# BAD: Explains what the loop does
for item in items:  # Loop through each item
    process(item)  # Process the item

# GOOD: No comment needed
if user is None:
    return False
return True

# GOOD: Explain non-obvious logic only
for item in items:
    # Process in reverse order to maintain dependency chain
    process(item)
```

---

## 3. Over-Documenting

Not every function needs extensive documentation.

### Trivial Functions

```python
# BAD: Over-documented for a simple function
def get_name(self) -> str:
    """Get the name of this object.

    This method returns the name attribute of the current instance.
    The name is a string that identifies this object.

    Args:
        self: The current instance.

    Returns:
        str: The name of this object as a string value.
    """
    return self.name

# GOOD: Let the signature speak
def get_name(self) -> str:
    """Return the object's display name."""
    return self.name

# BEST: If truly obvious, minimal or no docstring
@property
def name(self) -> str:
    return self._name
```

### Restating Types

```python
# BAD: Restates type annotations
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum as an integer.
    """
    return a + b

# GOOD: Add value beyond types
def add(a: int, b: int) -> int:
    """Add two integers with overflow protection.

    Raises:
        OverflowError: If the sum exceeds sys.maxsize.
    """
    return a + b

# BEST: For truly simple functions, minimal
def add(a: int, b: int) -> int:
    """Return the sum of a and b."""
    return a + b
```

---

## 4. Useless Section Markers

Comments that add structure but no information.

```python
# BAD: Empty section markers
class DataProcessor:
    # ============ CONSTANTS ============
    MAX_SIZE = 1000

    # ============ INITIALIZATION ============
    def __init__(self):
        pass

    # ============ PUBLIC METHODS ============
    def process(self):
        pass

    # ============ PRIVATE METHODS ============
    def _helper(self):
        pass

# GOOD: Let code organization speak for itself
class DataProcessor:
    MAX_SIZE = 1000

    def __init__(self):
        pass

    def process(self):
        pass

    def _helper(self):
        pass

# ACCEPTABLE: Meaningful section breaks in long functions
def complex_operation(data):
    # --- Validation phase ---
    # (actual meaningful comments about validation logic)

    # --- Transformation phase ---
    # (actual meaningful comments about transformation)

    # --- Output phase ---
    # (actual meaningful comments about output handling)
```

---

## 5. TODO Comments Without Context

```python
# BAD: Vague TODOs
# TODO: fix this
# TODO: refactor
# TODO: handle error
# FIXME

# BAD: Personal TODOs
# TODO(john): remember to update this

# GOOD: Actionable with context
# TODO: Add retry logic for transient network failures (see issue #45)
# TODO: Extract validation to separate module when adding new validators
# FIXME: Race condition when concurrent writes exceed pool size

# BEST: Link to issue tracker
# TODO(#123): Implement rate limiting before public release
```

---

## 6. Changelog in Docstrings

```python
# BAD: Embedded changelog
def calculate_score(metrics: dict) -> float:
    """Calculate aggregate score from metrics.

    Changelog:
        v2.0: Added weighted average calculation
        v1.5: Fixed rounding error
        v1.0: Initial implementation

    Args:
        metrics: Dictionary of metric values.
    """

# GOOD: Current behavior only
def calculate_score(metrics: dict) -> float:
    """Calculate weighted aggregate score from metrics.

    Uses exponential weighting to favor recent metrics.

    Args:
        metrics: Dictionary mapping metric names to values.

    Returns:
        Weighted average score between 0.0 and 1.0.
    """
```

---

## 7. Self-Evident Exceptions

```python
# BAD: Documents violations of the stated contract
def divide(a: int, b: int) -> float:
    """Divide a by b.

    Raises:
        ZeroDivisionError: If b is zero.
        TypeError: If a or b is not a number.
    """

# GOOD: Only document interface-relevant exceptions
def divide(a: int, b: int) -> float:
    """Divide a by b.

    Raises:
        ZeroDivisionError: If b is zero.
    """
    # TypeError is a contract violation, not part of interface
```

---

## 8. Placeholder Documentation

```python
# BAD: Obvious placeholders
def important_function():
    """TODO: Add docstring."""
    pass

def another_function():
    """Description goes here."""
    pass

def third_function():
    """..."""
    pass

# GOOD: Either document properly or omit
def important_function():
    """Perform the critical business operation."""
    pass

# Or for draft code, be explicit
def experimental_function():
    raise NotImplementedError("Draft implementation")
```

---

## Summary Checklist

Before committing documentation, verify:

- [ ] No references to changes, fixes, or previous behavior
- [ ] No narration of obvious code
- [ ] No restating of type annotations
- [ ] No empty section markers
- [ ] No vague or context-free TODOs
- [ ] No embedded changelogs
- [ ] No documentation of contract violations
- [ ] No placeholder text

**Remember**: Documentation describes the current state as authoritative truth.
