To resolve the issue, we need to ensure that `OUT_OF_SCOPE` is included as a termination reason in the `TerminationReason` class.

```python
class TerminationReason(Enum):
    AGENT_STOP = "AGENT_STOP"
    # other members...
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
```