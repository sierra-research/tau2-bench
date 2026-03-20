"""Integration tests for tau2 <-> A2A message translation."""

import json

import pytest

from tau2.a2a.translation import (
    A2A_AGENT_INSTRUCTION,
    a2a_to_tau2_assistant_message,
    format_system_context,
    format_tools_as_text,
    parse_a2a_tool_calls,
    tau2_to_a2a_message_content,
)
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.environment.tool import Tool

# Mark all tests in this module as mock-based (no real endpoints)
pytestmark = pytest.mark.a2a_mock


@pytest.fixture
def sample_tools():
    """Create sample tools for testing."""

    # Create a simple search_flights tool
    def search_flights(origin: str, destination: str, date: str) -> dict:
        """Search for available flights."""
        return {"flights": []}

    search_tool = Tool(search_flights)

    # Create a simple book_flight tool
    def book_flight(flight_id: str, passenger_info: dict) -> dict:
        """Book a specific flight."""
        return {"booking_id": "12345"}

    book_tool = Tool(book_flight)

    return [search_tool, book_tool]


def test_format_tools_as_text(sample_tools):
    """Test conversion of tau2 Tools to text description."""
    tool_text = format_tools_as_text(sample_tools)

    # Verify structure - format_tools_as_text only formats the tool list,
    # the usage instruction is added by tau2_to_a2a_message_content
    assert "<available_tools>" in tool_text
    assert "</available_tools>" in tool_text
    assert "search_flights" in tool_text
    assert "book_flight" in tool_text

    # Verify parameter descriptions
    assert "origin" in tool_text
    assert "destination" in tool_text
    assert "date" in tool_text
    assert "flight_id" in tool_text
    assert "passenger_info" in tool_text


def test_format_tools_as_text_empty():
    """Test format_tools_as_text with empty tool list."""
    tool_text = format_tools_as_text([])
    assert tool_text == ""


def test_format_system_context(sample_tools):
    """Test format_system_context matches LLMAgent system prompt structure."""
    domain_policy = "This is the test domain policy."
    context = format_system_context(domain_policy, sample_tools)

    # Verify it matches LLMAgent's system prompt structure
    assert "<instructions>" in context
    assert "</instructions>" in context
    assert "<policy>" in context
    assert "</policy>" in context
    assert A2A_AGENT_INSTRUCTION in context
    assert domain_policy in context

    # Should include tools
    assert "<available_tools>" in context
    assert "search_flights" in context
    assert "book_flight" in context

    # Should include tool usage instruction
    assert "To use a tool, respond with JSON" in context


def test_format_system_context_without_tools():
    """Test format_system_context without tools."""
    domain_policy = "This is the test domain policy."
    context = format_system_context(domain_policy, tools=None)

    # Should still have instructions and policy
    assert "<instructions>" in context
    assert "<policy>" in context
    assert domain_policy in context

    # Should not have tools section
    assert "<available_tools>" not in context


def test_tau2_to_a2a_first_message_includes_system_context(sample_tools):
    """Test that first message includes full system context like LLMAgent."""
    user_msg = UserMessage(
        role="user",
        content="I need help with a flight.",
    )
    domain_policy = "This is the test policy."

    # First message should include full system context
    content = tau2_to_a2a_message_content(
        user_msg,
        tools=sample_tools,
        domain_policy=domain_policy,
        is_first_message=True,
    )

    # Should include LLMAgent-style system context
    assert "<instructions>" in content
    assert "</instructions>" in content
    assert "<policy>" in content
    assert "</policy>" in content
    assert domain_policy in content
    assert A2A_AGENT_INSTRUCTION in content

    # Should include tools and user message
    assert "<available_tools>" in content
    assert "I need help with a flight." in content


def test_tau2_to_a2a_subsequent_message_no_policy(sample_tools):
    """Test that subsequent messages don't include full system context."""
    user_msg = UserMessage(
        role="user",
        content="Can you search for flights?",
    )
    domain_policy = "This is the test policy."

    # Subsequent message should NOT include policy/instructions
    content = tau2_to_a2a_message_content(
        user_msg,
        tools=sample_tools,
        domain_policy=domain_policy,
        is_first_message=False,
    )

    # Should NOT include system context
    assert "<instructions>" not in content
    assert "<policy>" not in content
    assert domain_policy not in content

    # Should still include tools for reminder
    assert "<available_tools>" in content
    assert "Can you search for flights?" in content


def test_tau2_to_a2a_user_message(sample_tools):
    """Test converting tau2 UserMessage to A2A content."""
    user_msg = UserMessage(
        role="user",
        content="I need to book a flight from SFO to JFK on December 15th.",
    )

    # Convert with tools
    content = tau2_to_a2a_message_content(user_msg, tools=sample_tools)

    # Should include original content and tool descriptions
    assert "I need to book a flight from SFO to JFK" in content
    assert "<available_tools>" in content
    assert "search_flights" in content
    assert "book_flight" in content


def test_tau2_to_a2a_user_message_without_tools():
    """Test converting tau2 UserMessage without tools."""
    user_msg = UserMessage(
        role="user",
        content="Hello, how can you help me?",
    )

    # Convert without tools
    content = tau2_to_a2a_message_content(user_msg, tools=None)

    # Should only include original content
    assert content == "Hello, how can you help me?"
    assert "<available_tools>" not in content


def test_tau2_to_a2a_assistant_message_text():
    """Test converting tau2 AssistantMessage with text content."""
    assistant_msg = AssistantMessage(
        role="assistant",
        content="I'll help you search for flights.",
        tool_calls=None,
    )

    content = tau2_to_a2a_message_content(assistant_msg)

    assert content == "I'll help you search for flights."


def test_tau2_to_a2a_assistant_message_tool_call():
    """Test converting tau2 AssistantMessage with tool call."""
    from tau2.data_model.message import ToolCall

    tool_call = ToolCall(
        id="call_123",
        name="search_flights",
        arguments={"origin": "SFO", "destination": "JFK", "date": "2025-12-15"},
        requestor="assistant",
    )

    assistant_msg = AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[tool_call],
    )

    content = tau2_to_a2a_message_content(assistant_msg)

    # Should be JSON with tool_call
    parsed = json.loads(content)
    assert "tool_call" in parsed
    assert parsed["tool_call"]["id"] == "call_123"
    assert parsed["tool_call"]["name"] == "search_flights"
    assert parsed["tool_call"]["arguments"]["origin"] == "SFO"


def test_tau2_to_a2a_tool_message():
    """Test converting tau2 ToolMessage to A2A content."""
    tool_msg = ToolMessage(
        id="call_123",
        role="tool",
        content='{"flights": [{"id": "AA123", "price": 350}]}',
        error=False,
        requestor="assistant",
    )

    content = tau2_to_a2a_message_content(tool_msg)

    # Should include tool result prefix and content
    assert "Tool result" in content
    assert "call_123" in content
    assert "AA123" in content


def test_tau2_to_a2a_tool_message_error():
    """Test converting tau2 ToolMessage with error."""
    tool_msg = ToolMessage(
        id="call_456",
        role="tool",
        content="Flight not found",
        error=True,
        requestor="assistant",
    )

    content = tau2_to_a2a_message_content(tool_msg)

    # Should include ERROR prefix
    assert "ERROR" in content
    assert "Flight not found" in content


def test_parse_a2a_tool_calls_single():
    """Test parsing single tool call from A2A response."""
    a2a_content = json.dumps(
        {
            "tool_call": {
                "name": "search_flights",
                "arguments": {
                    "origin": "SFO",
                    "destination": "JFK",
                    "date": "2025-12-15",
                },
            }
        }
    )

    tool_calls = parse_a2a_tool_calls(a2a_content)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "search_flights"
    assert tool_calls[0].arguments["origin"] == "SFO"
    assert tool_calls[0].requestor == "assistant"


def test_parse_a2a_tool_calls_multiple():
    """Test parsing multiple tool calls from A2A response."""
    a2a_content = json.dumps(
        {
            "tool_calls": [
                {
                    "tool_call": {
                        "name": "search_flights",
                        "arguments": {"origin": "SFO", "destination": "JFK"},
                    }
                },
                {
                    "tool_call": {
                        "name": "book_flight",
                        "arguments": {"flight_id": "AA123"},
                    }
                },
            ]
        }
    )

    tool_calls = parse_a2a_tool_calls(a2a_content)

    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0].name == "search_flights"
    assert tool_calls[1].name == "book_flight"


def test_parse_a2a_tool_calls_not_json():
    """Test parsing non-JSON content (regular text response)."""
    a2a_content = "I'll help you search for flights."

    tool_calls = parse_a2a_tool_calls(a2a_content)

    # Should return None for non-JSON content
    assert tool_calls is None


def test_parse_a2a_tool_calls_empty():
    """Test parsing empty content."""
    tool_calls = parse_a2a_tool_calls("")
    assert tool_calls is None

    tool_calls = parse_a2a_tool_calls(None)
    assert tool_calls is None


def test_parse_a2a_tool_calls_markdown_code_block():
    """Test parsing tool call wrapped in markdown code block."""
    # This is the format some LLMs use (e.g., vacation-rentals-agent)
    a2a_content = """```json
{"tool_call": {"name": "get_user_details", "arguments": {"user_id": "emeka_eze_5678"}}}
```"""

    tool_calls = parse_a2a_tool_calls(a2a_content)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_user_details"
    assert tool_calls[0].arguments["user_id"] == "emeka_eze_5678"
    assert tool_calls[0].requestor == "assistant"


def test_parse_a2a_tool_calls_markdown_code_block_no_language():
    """Test parsing tool call wrapped in markdown code block without language specifier."""
    a2a_content = """```
{"tool_call": {"name": "search_flights", "arguments": {"origin": "SFO"}}}
```"""

    tool_calls = parse_a2a_tool_calls(a2a_content)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "search_flights"
    assert tool_calls[0].arguments["origin"] == "SFO"


def test_a2a_to_tau2_assistant_message_text():
    """Test converting A2A text response to tau2 AssistantMessage."""
    a2a_content = "I found 5 flights for you."

    assistant_msg = a2a_to_tau2_assistant_message(a2a_content)

    assert isinstance(assistant_msg, AssistantMessage)
    assert assistant_msg.role == "assistant"
    assert assistant_msg.content == "I found 5 flights for you."
    assert assistant_msg.tool_calls is None or len(assistant_msg.tool_calls) == 0


def test_a2a_to_tau2_assistant_message_tool_call():
    """Test converting A2A tool call response to tau2 AssistantMessage."""
    a2a_content = json.dumps(
        {
            "tool_call": {
                "name": "search_flights",
                "arguments": {"origin": "SFO", "destination": "JFK"},
            }
        }
    )

    assistant_msg = a2a_to_tau2_assistant_message(a2a_content)

    assert isinstance(assistant_msg, AssistantMessage)
    assert assistant_msg.role == "assistant"
    assert assistant_msg.content is None
    assert assistant_msg.tool_calls is not None
    assert len(assistant_msg.tool_calls) == 1
    assert assistant_msg.tool_calls[0].name == "search_flights"


class TestTranslationEdgeCases:
    """Edge cases for translation functions — empty inputs, missing fields."""

    def test_a2a_to_tau2_empty_string_uses_fallback(self):
        """Empty string content triggers fallback message."""
        msg = a2a_to_tau2_assistant_message("")

        assert "apologize" in msg.content.lower()
        assert msg.tool_calls is None

    def test_a2a_to_tau2_whitespace_only_uses_fallback(self):
        """Whitespace-only content triggers fallback message."""
        msg = a2a_to_tau2_assistant_message("   \n\t  ")

        assert "apologize" in msg.content.lower()

    def test_a2a_to_tau2_none_uses_fallback(self):
        """None content triggers fallback message without crashing."""
        msg = a2a_to_tau2_assistant_message(None)

        assert "apologize" in msg.content.lower()

    def test_parse_tool_call_missing_name_raises(self):
        """Malformed tool call without 'name' raises A2AMessageError."""
        from tau2.a2a.exceptions import A2AMessageError

        content = json.dumps({"tool_call": {"arguments": {"x": 1}}})

        with pytest.raises(A2AMessageError, match="Invalid tool call format"):
            parse_a2a_tool_calls(content)

    def test_parse_tool_call_missing_arguments_raises(self):
        """Malformed tool call without 'arguments' raises A2AMessageError."""
        from tau2.a2a.exceptions import A2AMessageError

        content = json.dumps({"tool_call": {"name": "foo"}})

        with pytest.raises(A2AMessageError, match="Invalid tool call format"):
            parse_a2a_tool_calls(content)

    def test_tau2_to_a2a_assistant_no_text_no_tools_returns_empty(self):
        """AssistantMessage with neither text nor tool_calls returns empty string."""
        msg = AssistantMessage(role="assistant", content=None, tool_calls=None)

        result = tau2_to_a2a_message_content(msg)

        assert result == ""

    def test_tau2_to_a2a_assistant_empty_content_returns_empty(self):
        """AssistantMessage with empty string content returns empty string."""
        msg = AssistantMessage(role="assistant", content="", tool_calls=None)

        result = tau2_to_a2a_message_content(msg)

        assert result == ""


def test_roundtrip_translation_preserves_content():
    """Test that roundtrip translation preserves message content."""
    # User message roundtrip
    original_user = UserMessage(role="user", content="Search for flights to NYC")
    a2a_content = tau2_to_a2a_message_content(original_user)
    assert "Search for flights to NYC" in a2a_content

    # Assistant message roundtrip (text)
    a2a_text_response = "I found 3 flights."
    tau2_assistant = a2a_to_tau2_assistant_message(a2a_text_response)
    assert tau2_assistant.content == "I found 3 flights."

    # Assistant message roundtrip (tool call)
    from tau2.data_model.message import ToolCall

    original_tool_call = ToolCall(
        id="call_1",
        name="search_flights",
        arguments={"origin": "SFO"},
        requestor="assistant",
    )
    original_assistant = AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[original_tool_call],
    )

    # Convert to A2A and back
    a2a_tool_content = tau2_to_a2a_message_content(original_assistant)
    recovered_assistant = a2a_to_tau2_assistant_message(a2a_tool_content)

    # Verify tool call preserved
    assert recovered_assistant.tool_calls is not None
    assert len(recovered_assistant.tool_calls) == 1
    assert recovered_assistant.tool_calls[0].name == "search_flights"
    assert recovered_assistant.tool_calls[0].arguments["origin"] == "SFO"
