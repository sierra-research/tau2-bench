#!/usr/bin/env python3
"""
Example script demonstrating tool call error tracking.

This shows how to use the new error tracking features added to llm_utils.py:
1. Errors are automatically tracked during tool calls
2. Errors are automatically saved on program exit (with timestamp)
3. Call save_tool_call_error_analysis() to manually save
4. Call get_tool_call_error_stats() to get current stats
5. Call reset_tool_call_error_tracking() to reset counters
"""

from tau2.utils.llm_utils import (
    save_tool_call_error_analysis,
    get_tool_call_error_stats,
    reset_tool_call_error_tracking,
)

# Example usage in your code:
# After running your LLM interactions with tool calls...

# 1. Get current statistics
stats = get_tool_call_error_stats()
print(f"Total tool call errors: {stats['total']}")
print(f"Error breakdown: {stats['counts']}")

# 2. Save detailed analysis to file (optional - auto-saves on exit)
# Uses timestamped filename by default: error_call_analysis_{timestamp}.txt
save_tool_call_error_analysis()

# Or specify a custom filename:
# save_tool_call_error_analysis("my_custom_errors.txt")

# 3. If you want to reset and start tracking fresh
# reset_tool_call_error_tracking()

print("\n✅ Error analysis saved!")
print("Note: Errors are also automatically saved when the program exits.")
