"""
Tests for audio chunking functionality in streaming mixins.

This module tests AudioChunkingMixin to ensure:
1. Chunking is done correctly (right number of chunks, proper chunk properties)
2. Chunking and merging are inverse operations

Note: TextChunkingMixin tests have been moved to
src/experiments/tau_voice/tests/test_text_chunking.py
"""

import base64

import pytest

from tau2.agent.base.streaming import AudioChunkingMixin
from tau2.data_model.audio import AudioEncoding, AudioFormat
from tau2.data_model.message import AssistantMessage, UserMessage

# ============================================================================
# Audio Chunking Tests
# ============================================================================


class TestAudioChunking:
    """Tests for AudioChunkingMixin."""

    @pytest.fixture
    def audio_chunker(self):
        """Create an audio chunker."""

        class SimpleAudioChunker(AudioChunkingMixin):
            """Minimal implementation for testing."""

            def _next_turn_taking_action(self, state):
                return "generate_message"

            def _should_respond_to_chunk(self, incoming_chunk, state):
                return True

            def speech_detection(self, incoming_chunk):
                return True

            def _perform_turn_taking_action(self, state, action):
                return None, state

            def _process_tool_result(self, tool_result, state):
                return None, state

            def _emit_waiting_chunk(self, state):
                return None, state

        # chunk_size is in samples, e.g., 50 samples per chunk
        return SimpleAudioChunker(chunk_size=50)

    @pytest.fixture
    def audio_message(self):
        """Create a sample audio message for testing."""
        # Create fake audio data: 120 samples at 16-bit (2 bytes per sample)
        # Total: 240 bytes
        sample_rate = 16000
        num_samples = 120
        audio_bytes = b"\x00\x01" * num_samples  # 240 bytes total

        audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
        )

        return AssistantMessage(
            role="assistant",
            content="Hello world",
            is_audio=True,
            audio_content=base64.b64encode(audio_bytes).decode("utf-8"),
            audio_format=audio_format,
            audio_script_gold="Hello world",
            cost=0.02,
            usage={"tokens": 20},
        )

    def test_audio_chunking_correct_number_of_chunks(
        self, audio_chunker, audio_message
    ):
        """Test that audio chunking produces the correct number of chunks."""
        chunks = audio_chunker._create_chunk_messages(audio_message)

        # 120 samples / 50 samples per chunk = 3 chunks (last one padded)
        expected_chunks = 3
        assert len(chunks) == expected_chunks

    def test_audio_chunks_have_same_number_of_samples(
        self, audio_chunker, audio_message
    ):
        """Test that all audio chunks have exactly chunk_size samples (with padding)."""
        chunks = audio_chunker._create_chunk_messages(audio_message)

        for chunk in chunks:
            # Decode the audio content
            audio_bytes = base64.b64decode(chunk.audio_content)
            bytes_per_sample = chunk.audio_format.bytes_per_sample
            num_samples = len(audio_bytes) // bytes_per_sample

            # All chunks should have exactly chunk_size samples
            assert num_samples == audio_chunker.chunk_size

    def test_audio_chunks_have_correct_metadata(self, audio_chunker, audio_message):
        """Test that audio chunks have correct chunk_id and is_final_chunk metadata."""
        chunks = audio_chunker._create_chunk_messages(audio_message)

        # Check chunk IDs are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == i
            assert chunk.is_final_chunk == (i == len(chunks) - 1)
            assert chunk.is_audio is True

    def test_audio_chunks_preserve_format(self, audio_chunker, audio_message):
        """Test that audio format is preserved across chunks."""
        chunks = audio_chunker._create_chunk_messages(audio_message)

        for chunk in chunks:
            assert (
                chunk.audio_format.sample_rate == audio_message.audio_format.sample_rate
            )
            assert chunk.audio_format.channels == audio_message.audio_format.channels
            assert (
                chunk.audio_format.bytes_per_sample
                == audio_message.audio_format.bytes_per_sample
            )

    def test_audio_chunks_preserve_cost_in_first_chunk_only(
        self, audio_chunker, audio_message
    ):
        """Test that cost/usage metadata is only in the first chunk."""
        chunks = audio_chunker._create_chunk_messages(audio_message)

        # First chunk should have cost and usage
        assert chunks[0].cost == 0.02
        assert chunks[0].usage == {"tokens": 20}

        # Other chunks should have zero cost and None usage
        for chunk in chunks[1:]:
            assert chunk.cost == 0.0
            assert chunk.usage is None

    def test_audio_chunking_and_merging_are_inverse_with_padding(
        self, audio_chunker, audio_message
    ):
        """Test that chunking then merging preserves audio structure.

        Verifies that:
        1. Chunks can be merged back together
        2. The merged length equals ceil(original_samples / chunk_size) * chunk_size
        3. The bulk of the original audio is preserved (merge may apply crossfade
           at chunk boundaries, so the last few samples can differ)
        4. The last sample of the merged audio is silent (fade reaches zero)
        """
        original_audio_bytes = audio_message.get_audio_bytes()
        original_num_samples = (
            len(original_audio_bytes) // audio_message.audio_format.bytes_per_sample
        )

        # Chunk the message
        chunks = audio_chunker._create_chunk_messages(audio_message)

        # Verify we have multiple chunks
        assert len(chunks) > 1

        # Verify each chunk has the correct size (chunk_size samples)
        for chunk in chunks:
            chunk_bytes = base64.b64decode(chunk.audio_content)
            chunk_samples = len(chunk_bytes) // chunk.audio_format.bytes_per_sample
            assert chunk_samples == audio_chunker.chunk_size

        # Merge the chunks back
        merged = AssistantMessage.merge_chunks(chunks)
        merged_audio_bytes = merged.get_audio_bytes()

        # Calculate expected merged size (with padding in last chunk)
        import math

        expected_num_samples = (
            math.ceil(original_num_samples / audio_chunker.chunk_size)
            * audio_chunker.chunk_size
        )
        expected_bytes = (
            expected_num_samples * audio_message.audio_format.bytes_per_sample
        )

        # Verify merged audio has the expected length
        assert len(merged_audio_bytes) == expected_bytes

        # The bulk of the original audio should be preserved.
        # Merging may apply crossfade at chunk boundaries, so the final
        # samples of the original region can be modified by the fade-out.
        # Verify at least the first 80% of the original audio is identical.
        safe_prefix_len = int(len(original_audio_bytes) * 0.8)
        # Align to sample boundary
        bytes_per_sample = audio_message.audio_format.bytes_per_sample
        safe_prefix_len = (safe_prefix_len // bytes_per_sample) * bytes_per_sample
        assert (
            merged_audio_bytes[:safe_prefix_len]
            == original_audio_bytes[:safe_prefix_len]
        )

        # The very last sample of the merged audio should be silent (zero)
        # because any fade should reach zero by the end of the padding.
        last_sample = merged_audio_bytes[-bytes_per_sample:]
        assert last_sample == b"\x00" * bytes_per_sample, (
            "Last sample should be silent (zero) after fade-out"
        )

        # Check that properties are preserved
        assert merged.role == audio_message.role
        assert merged.is_audio is True
        assert merged.audio_format.sample_rate == audio_message.audio_format.sample_rate
        assert merged.audio_format.channels == audio_message.audio_format.channels
        assert merged.audio_script_gold is not None

    def test_audio_content_text_distribution(self, audio_chunker):
        """Test that text content (audio_script_gold) is distributed evenly across chunks."""
        # Create audio with 100 samples and text of 10 characters
        sample_rate = 16000
        num_samples = 100
        audio_bytes = b"\x00\x01" * num_samples

        audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
        )

        text_content = "0123456789"  # 10 characters
        message = AssistantMessage(
            role="assistant",
            content=text_content,
            is_audio=True,
            audio_content=base64.b64encode(audio_bytes).decode("utf-8"),
            audio_format=audio_format,
            audio_script_gold=text_content,
        )

        # Chunk with 50 samples per chunk -> 2 chunks
        chunks = audio_chunker._create_chunk_messages(message)

        # Collect all text content from chunks
        merged_text = "".join(chunk.content or "" for chunk in chunks)

        # Text should be distributed and recoverable
        assert merged_text == text_content

    def test_audio_content_text_distribution_fewer_chars_than_chunks(
        self, audio_chunker
    ):
        """Test text distribution when there are fewer characters than chunks."""
        # Create audio with 200 samples -> 4 chunks (50 samples each)
        sample_rate = 16000
        num_samples = 200
        audio_bytes = b"\x00\x01" * num_samples

        audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
        )

        text_content = "Hi"  # Only 2 characters, but 4 chunks
        message = AssistantMessage(
            role="assistant",
            content=text_content,
            is_audio=True,
            audio_content=base64.b64encode(audio_bytes).decode("utf-8"),
            audio_format=audio_format,
            audio_script_gold=text_content,
        )

        chunks = audio_chunker._create_chunk_messages(message)

        # Should have 4 chunks
        assert len(chunks) == 4

        # Collect all text content from chunks
        merged_text = "".join(chunk.content or "" for chunk in chunks)

        # Text should be interspersed and recoverable
        assert merged_text == text_content

        # Some chunks should have empty content (interspersed)
        empty_chunks = sum(1 for chunk in chunks if not chunk.content)
        assert empty_chunks == 2  # 2 chars distributed across 4 chunks -> 2 empty

    def test_audio_chunking_exact_multiple(self, audio_chunker):
        """Test audio chunking when samples are an exact multiple of chunk_size."""
        # Create audio with exactly 100 samples (2 chunks of 50)
        sample_rate = 16000
        num_samples = 100
        audio_bytes = b"\x00\x01" * num_samples

        audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
        )

        message = AssistantMessage(
            role="assistant",
            content="Test",
            is_audio=True,
            audio_content=base64.b64encode(audio_bytes).decode("utf-8"),
            audio_format=audio_format,
            audio_script_gold="Test",
        )

        chunks = audio_chunker._create_chunk_messages(message)

        # Should have exactly 2 chunks, no padding needed
        assert len(chunks) == 2

        for chunk in chunks:
            audio_bytes_chunk = base64.b64decode(chunk.audio_content)
            bytes_per_sample = chunk.audio_format.bytes_per_sample
            num_samples_chunk = len(audio_bytes_chunk) // bytes_per_sample
            assert num_samples_chunk == audio_chunker.chunk_size


# ============================================================================
# Content Chunks Distribution Tests (for _create_content_chunks)
# ============================================================================


class TestContentChunksDistribution:
    """Test the _create_content_chunks helper method for even distribution."""

    @pytest.fixture
    def audio_chunker(self):
        """Create an audio chunker to access _create_content_chunks."""

        class SimpleAudioChunker(AudioChunkingMixin):
            def _next_turn_taking_action(self, state):
                return "generate_message"

            def _should_respond_to_chunk(self, incoming_chunk, state):
                return True

            def speech_detection(self, incoming_chunk):
                return True

            def _perform_turn_taking_action(self, state, action):
                return None, state

            def _process_tool_result(self, tool_result, state):
                return None, state

            def _emit_waiting_chunk(self, state):
                return None, state

        return SimpleAudioChunker(chunk_size=50)

    def test_more_chars_than_chunks(self, audio_chunker):
        """Test character distribution when there are more characters than chunks."""
        content = "0123456789"  # 10 characters
        num_chunks = 3

        chunks = audio_chunker._create_content_chunks(content, num_chunks)

        # Should have 3 chunks
        assert len(chunks) == 3

        # Merged content should equal original
        assert "".join(chunks) == content

        # Check distribution: 10 chars / 3 chunks = 3,3,4 or similar
        chunk_lengths = [len(chunk) for chunk in chunks]
        assert sum(chunk_lengths) == len(content)
        # All chunks should have similar length (within 1 character)
        assert max(chunk_lengths) - min(chunk_lengths) <= 1

    def test_fewer_chars_than_chunks(self, audio_chunker):
        """Test character distribution when there are fewer characters than chunks."""
        content = "Hi"  # 2 characters
        num_chunks = 5

        chunks = audio_chunker._create_content_chunks(content, num_chunks)

        # Should have 5 chunks
        assert len(chunks) == 5

        # Merged content should equal original
        assert "".join(chunks) == content

        # Should have 2 non-empty and 3 empty chunks (interspersed)
        non_empty = [chunk for chunk in chunks if chunk]
        empty = [chunk for chunk in chunks if not chunk]
        assert len(non_empty) == 2
        assert len(empty) == 3

        # Non-empty chunks should be interspersed (not consecutive)
        # e.g., ["H", "", "i", "", ""] or similar pattern
        non_empty_indices = [i for i, chunk in enumerate(chunks) if chunk]
        # Check that they're not consecutive
        if len(non_empty_indices) > 1:
            assert non_empty_indices[1] - non_empty_indices[0] > 1

    def test_equal_chars_and_chunks(self, audio_chunker):
        """Test character distribution when chars equal chunks."""
        content = "12345"  # 5 characters
        num_chunks = 5

        chunks = audio_chunker._create_content_chunks(content, num_chunks)

        # Should have 5 chunks, each with 1 character
        assert len(chunks) == 5
        assert all(len(chunk) == 1 for chunk in chunks)
        assert "".join(chunks) == content

    def test_empty_content(self, audio_chunker):
        """Test character distribution with empty content."""
        content = ""
        num_chunks = 3

        chunks = audio_chunker._create_content_chunks(content, num_chunks)

        # Should have 3 chunks, all empty
        assert len(chunks) == 3
        assert all(chunk == "" for chunk in chunks)

    def test_single_chunk(self, audio_chunker):
        """Test character distribution with single chunk."""
        content = "Hello world"
        num_chunks = 1

        chunks = audio_chunker._create_content_chunks(content, num_chunks)

        # Should have 1 chunk with all content
        assert len(chunks) == 1
        assert chunks[0] == content


# ============================================================================
# Message Merging Tests
# ============================================================================


class TestMessageMerging:
    """Tests for the merge_chunks method."""

    def test_merge_text_chunks_basic(self):
        """Test basic text chunk merging (direct concatenation without separator)."""
        chunks = [
            UserMessage(
                role="user", content="Hello ", chunk_id=0, is_final_chunk=False
            ),
            UserMessage(role="user", content="world", chunk_id=1, is_final_chunk=True),
        ]

        merged = UserMessage.merge_chunks(chunks)

        assert merged.role == "user"
        assert merged.content == "Hello world"
        assert merged.is_audio is False

    def test_merge_text_chunks_with_empty(self):
        """Test merging text chunks with empty content (direct concatenation)."""
        chunks = [
            UserMessage(role="user", content="Hello", chunk_id=0, is_final_chunk=False),
            UserMessage(role="user", content="", chunk_id=1, is_final_chunk=False),
            UserMessage(role="user", content="world", chunk_id=2, is_final_chunk=True),
        ]

        merged = UserMessage.merge_chunks(chunks)

        # Direct concatenation - empty chunk contributes nothing
        assert merged.content == "Helloworld"

    def test_merge_single_chunk(self):
        """Test merging a single chunk."""
        chunks = [
            UserMessage(role="user", content="Hello", chunk_id=0, is_final_chunk=True),
        ]

        merged = UserMessage.merge_chunks(chunks)

        assert merged.content == "Hello"
        assert merged.role == "user"

    def test_merge_audio_chunks_basic(self):
        """Test basic audio chunk merging."""
        sample_rate = 16000
        audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
        )

        # Create chunks with different audio data
        chunk1_bytes = b"\x00\x01" * 10  # 20 bytes
        chunk2_bytes = b"\x02\x03" * 10  # 20 bytes

        # Each chunk has full template with 'active' attribute (as produced by the chunker)
        chunks = [
            AssistantMessage(
                role="assistant",
                content="Hel",
                is_audio=True,
                audio_content=base64.b64encode(chunk1_bytes).decode("utf-8"),
                audio_format=audio_format,
                audio_script_gold='<message uuid="test-uuid" active="0"><chunk id=0>Hel</chunk><chunk id=1>lo</chunk></message>',
                chunk_id=0,
                is_final_chunk=False,
            ),
            AssistantMessage(
                role="assistant",
                content="lo",
                is_audio=True,
                audio_content=base64.b64encode(chunk2_bytes).decode("utf-8"),
                audio_format=audio_format,
                audio_script_gold='<message uuid="test-uuid" active="1"><chunk id=0>Hel</chunk><chunk id=1>lo</chunk></message>',
                chunk_id=1,
                is_final_chunk=True,
            ),
        ]

        merged = AssistantMessage.merge_chunks(chunks)

        # Verify merged audio is concatenation of chunk bytes
        merged_bytes = merged.get_audio_bytes()
        expected_bytes = chunk1_bytes + chunk2_bytes
        assert merged_bytes == expected_bytes

        # Verify audio_script_gold has both chunks tagged (both were received)
        assert "<chunk id=0>Hel</chunk>" in merged.audio_script_gold
        assert "<chunk id=1>lo</chunk>" in merged.audio_script_gold

    def test_merge_validates_same_type(self):
        """Test that merge_chunks validates all chunks are of the same type.

        Note: Type checking happens before role checking, so mixing UserMessage
        and AssistantMessage will fail on type validation first.
        """
        chunks = [
            UserMessage(role="user", content="Hello", chunk_id=0, is_final_chunk=False),
            AssistantMessage(
                role="assistant", content="world", chunk_id=1, is_final_chunk=True
            ),
        ]

        with pytest.raises(ValueError, match="All chunks must be of the same type"):
            UserMessage.merge_chunks(chunks)

    def test_merge_validates_no_tool_calls(self):
        """Test that merge_chunks rejects chunks with tool calls."""
        from tau2.data_model.message import ToolCall

        chunks = [
            AssistantMessage(
                role="assistant",
                content="Hello",
                chunk_id=0,
                is_final_chunk=False,
            ),
            AssistantMessage(
                role="assistant",
                content="world",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="test_function",
                        arguments={},
                        requestor="assistant",
                    )
                ],
                chunk_id=1,
                is_final_chunk=True,
            ),
        ]

        with pytest.raises(
            ValueError, match="Cannot merge chunks that contain tool calls"
        ):
            AssistantMessage.merge_chunks(chunks)

    def test_merge_validates_all_audio_or_none(self):
        """Test that merge_chunks validates chunks are all audio or all text."""
        sample_rate = 16000
        audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
        )

        chunks = [
            AssistantMessage(
                role="assistant",
                content="Hello",
                is_audio=False,
                chunk_id=0,
                is_final_chunk=False,
            ),
            AssistantMessage(
                role="assistant",
                content="world",
                is_audio=True,
                audio_content=base64.b64encode(b"\x00\x01" * 10).decode("utf-8"),
                audio_format=audio_format,
                chunk_id=1,
                is_final_chunk=True,
            ),
        ]

        with pytest.raises(
            ValueError, match="All chunks must be either audio or non-audio"
        ):
            AssistantMessage.merge_chunks(chunks)

    def test_merge_validates_same_audio_format(self):
        """Test that merge_chunks validates all audio chunks have same format."""
        audio_format_1 = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=16000,
        )
        audio_format_2 = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=8000,  # Different sample rate
        )

        chunks = [
            AssistantMessage(
                role="assistant",
                content="Hello",
                is_audio=True,
                audio_content=base64.b64encode(b"\x00\x01" * 10).decode("utf-8"),
                audio_format=audio_format_1,
                chunk_id=0,
                is_final_chunk=False,
            ),
            AssistantMessage(
                role="assistant",
                content="world",
                is_audio=True,
                audio_content=base64.b64encode(b"\x00\x01" * 10).decode("utf-8"),
                audio_format=audio_format_2,
                chunk_id=1,
                is_final_chunk=True,
            ),
        ]

        with pytest.raises(
            ValueError, match="All audio chunks must have the same audio format"
        ):
            AssistantMessage.merge_chunks(chunks)

    def test_merge_rejects_empty_list(self):
        """Test that merge_chunks rejects empty chunk list."""
        with pytest.raises(ValueError, match="Cannot merge empty list of chunks"):
            UserMessage.merge_chunks([])

    def test_merge_validates_same_role_within_type(self):
        """Test that merge_chunks validates all chunks have the same role.

        This tests the role validation that happens after type validation.
        """
        # Create two UserMessages but with mismatched role strings
        # We need to bypass pydantic validation, so we'll modify after creation
        chunk1 = UserMessage(
            role="user", content="Hello", chunk_id=0, is_final_chunk=False
        )
        chunk2 = UserMessage(
            role="user", content="world", chunk_id=1, is_final_chunk=True
        )

        # Manually change the role to simulate a mismatch (bypassing pydantic)
        # In practice this shouldn't happen, but the validation exists to catch it
        chunk2.role = "other_user"  # type: ignore

        chunks = [chunk1, chunk2]

        with pytest.raises(ValueError, match="All chunks must be from the same role"):
            UserMessage.merge_chunks(chunks)

    def test_merge_preserves_metadata_semantics(self):
        """Test that merge doesn't copy chunk-specific metadata to merged message."""
        chunks = [
            UserMessage(
                role="user",
                content="Hello",
                chunk_id=0,
                is_final_chunk=False,
            ),
            UserMessage(
                role="user",
                content="world",
                chunk_id=1,
                is_final_chunk=True,
            ),
        ]

        merged = UserMessage.merge_chunks(chunks)

        # Merged message should not have chunk metadata
        assert merged.chunk_id is None
        assert merged.is_final_chunk is True  # Default value

    def test_merge_audio_with_none_content(self):
        """Test merging audio chunks where some have None audio_content."""
        audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=16000,
        )

        chunks = [
            AssistantMessage(
                role="assistant",
                content="Hello",
                is_audio=True,
                audio_content=base64.b64encode(b"\x00\x01" * 10).decode("utf-8"),
                audio_format=audio_format,
                chunk_id=0,
                is_final_chunk=False,
            ),
            AssistantMessage(
                role="assistant",
                content="world",
                is_audio=True,
                audio_content=None,  # No audio content
                audio_format=audio_format,
                chunk_id=1,
                is_final_chunk=True,
            ),
        ]

        merged = AssistantMessage.merge_chunks(chunks)

        # Should handle None gracefully (treated as empty)
        merged_bytes = merged.get_audio_bytes()
        assert merged_bytes == b"\x00\x01" * 10


# ============================================================================
# Audio Script Gold Chunking and Merging Tests
# ============================================================================


class TestAudioScriptGoldChunking:
    """Tests for audio_script_gold marking during chunking."""

    @pytest.fixture
    def audio_chunker(self):
        """Create an audio chunker."""

        class SimpleAudioChunker(AudioChunkingMixin):
            def _next_turn_taking_action(self, state):
                return "generate_message"

            def _should_respond_to_chunk(self, incoming_chunk, state):
                return True

            def speech_detection(self, incoming_chunk):
                return True

            def _perform_turn_taking_action(self, state, action):
                return None, state

            def _process_tool_result(self, tool_result, state):
                return None, state

            def _emit_waiting_chunk(self, state):
                return None, state

        return SimpleAudioChunker(chunk_size=50)

    @pytest.fixture
    def audio_message(self):
        """Create a sample audio message for testing."""
        sample_rate = 16000
        num_samples = 100  # 2 chunks of 50 samples
        audio_bytes = b"\x00\x01" * num_samples

        audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
        )

        return AssistantMessage(
            role="assistant",
            content="Hello world",
            is_audio=True,
            audio_content=base64.b64encode(audio_bytes).decode("utf-8"),
            audio_format=audio_format,
            audio_script_gold="Hello world",
        )

    def test_chunks_have_uuid_marked_audio_script_gold(
        self, audio_chunker, audio_message
    ):
        """Test that all chunks have audio_script_gold with UUID and chunk markers."""

        chunks = audio_chunker._create_chunk_messages(audio_message)

        for chunk in chunks:
            # Each chunk should have the marked format
            assert chunk.audio_script_gold is not None
            assert '<message uuid="' in chunk.audio_script_gold
            assert "</message>" in chunk.audio_script_gold
            assert "<chunk id=" in chunk.audio_script_gold

    def test_each_chunk_has_unique_audio_script_gold(
        self, audio_chunker, audio_message
    ):
        """Test that each chunk has a different audio_script_gold (only tags itself)."""
        chunks = audio_chunker._create_chunk_messages(audio_message)

        # Each chunk should have a different audio_script_gold
        # because each only tags its own chunk
        script_golds = [chunk.audio_script_gold for chunk in chunks]
        assert len(set(script_golds)) == len(chunks)

        # But all should have the same UUID
        from tau2.agent.base.streaming_utils import extract_message_uuid

        uuids = [extract_message_uuid(sg) for sg in script_golds]
        assert len(set(uuids)) == 1

    def test_each_chunk_audio_script_gold_has_correct_active(
        self, audio_chunker, audio_message
    ):
        """Test that each chunk's audio_script_gold has correct 'active' attribute."""
        from tau2.agent.base.streaming_utils import extract_active_chunk_ids

        chunks = audio_chunker._create_chunk_messages(audio_message)

        for i, chunk in enumerate(chunks):
            # Each chunk should have its own chunk_id in the active attribute
            active_ids = extract_active_chunk_ids(chunk.audio_script_gold)
            assert active_ids == {i}

    def test_audio_script_gold_text_matches_original(
        self, audio_chunker, audio_message
    ):
        """Test that text content in audio_script_gold matches original."""
        import re

        chunks = audio_chunker._create_chunk_messages(audio_message)

        script_gold = chunks[0].audio_script_gold

        # Extract text content (strip all tags)
        text_only = re.sub(r"<[^>]+>", "", script_gold)
        assert text_only == "Hello world"


class TestAudioScriptGoldMerging:
    """Tests for merge_audio_script_gold utility function."""

    def test_merge_single_message_all_chunks_received(self):
        """Test merging when all chunks of a single message are received."""
        from tau2.agent.base.streaming_utils import merge_audio_script_gold

        # Simulate all chunks received (all have <chunk> tags)
        script_gold = '<message uuid="abc-123"><chunk id=0>Hello </chunk><chunk id=1>world</chunk></message>'

        result = merge_audio_script_gold([script_gold, script_gold])

        # Should keep all chunk tags (both were received)
        assert result == script_gold

    def test_merge_single_message_partial_chunks_received(self):
        """Test merging when only some chunks were received."""
        from tau2.agent.base.streaming_utils import merge_audio_script_gold

        # Only chunk 0 was received (has tag), chunk 1 not received (plain text)
        script_gold = (
            '<message uuid="abc-123"><chunk id=0>Hello </chunk>world</message>'
        )

        result = merge_audio_script_gold([script_gold])

        # Should preserve the structure - chunk 0 tagged, chunk 1 plain
        assert "<chunk id=0>Hello </chunk>" in result
        assert "<chunk id=1>" not in result
        assert "world" in result

    def test_merge_combines_received_chunks_from_multiple_inputs(self):
        """Test that merging unions received chunks from multiple inputs."""
        from tau2.agent.base.streaming_utils import merge_audio_script_gold

        # Each chunk has full template with 'active' attribute (as produced by chunking)
        chunk0 = '<message uuid="abc-123" active="0"><chunk id=0>Hello </chunk><chunk id=1>world </chunk><chunk id=2>!</chunk></message>'
        chunk1 = '<message uuid="abc-123" active="1"><chunk id=0>Hello </chunk><chunk id=1>world </chunk><chunk id=2>!</chunk></message>'
        chunk2 = '<message uuid="abc-123" active="2"><chunk id=0>Hello </chunk><chunk id=1>world </chunk><chunk id=2>!</chunk></message>'

        result = merge_audio_script_gold([chunk0, chunk1, chunk2])

        # Should have all chunks tagged
        assert "<chunk id=0>Hello </chunk>" in result
        assert "<chunk id=1>world </chunk>" in result
        assert "<chunk id=2>!</chunk>" in result

    def test_merge_multiple_messages(self):
        """Test merging chunks from multiple different messages."""
        from tau2.agent.base.streaming_utils import merge_audio_script_gold

        msg1 = '<message uuid="uuid-1"><chunk id=0>Hello</chunk></message>'
        msg2 = '<message uuid="uuid-2"><chunk id=0>World</chunk></message>'

        result = merge_audio_script_gold([msg1, msg2])

        # Should have both messages
        assert '<message uuid="uuid-1">' in result
        assert '<message uuid="uuid-2">' in result
        assert "Hello" in result
        assert "World" in result

    def test_merge_preserves_message_order(self):
        """Test that merging preserves the order messages were first seen."""
        from tau2.agent.base.streaming_utils import merge_audio_script_gold

        msg1 = '<message uuid="first"><chunk id=0>A</chunk></message>'
        msg2 = '<message uuid="second"><chunk id=0>B</chunk></message>'

        result = merge_audio_script_gold([msg1, msg2])

        # First message should appear before second
        first_pos = result.find('uuid="first"')
        second_pos = result.find('uuid="second"')
        assert first_pos < second_pos

    def test_merge_handles_none_values(self):
        """Test that merge handles None values in input list."""
        from tau2.agent.base.streaming_utils import merge_audio_script_gold

        msg = '<message uuid="abc"><chunk id=0>Hello</chunk></message>'

        result = merge_audio_script_gold([None, msg, None])

        assert result == msg

    def test_merge_returns_none_for_empty_input(self):
        """Test that merge returns None for empty or all-None input."""
        from tau2.agent.base.streaming_utils import merge_audio_script_gold

        assert merge_audio_script_gold([]) is None
        assert merge_audio_script_gold([None, None]) is None

    def test_merge_handles_already_merged_results(self):
        """Test that merging already-merged results works correctly."""
        from tau2.agent.base.streaming_utils import (
            extract_active_chunk_ids,
            merge_audio_script_gold,
        )

        # Simulate: first we received chunks 0 and 2 (with active attribute)
        chunk0 = '<message uuid="abc" active="0"><chunk id=0>A</chunk><chunk id=1>B</chunk><chunk id=2>C</chunk></message>'
        chunk2 = '<message uuid="abc" active="2"><chunk id=0>A</chunk><chunk id=1>B</chunk><chunk id=2>C</chunk></message>'
        first_merge = merge_audio_script_gold([chunk0, chunk2])

        # First merge should have active="0,2" (chunks 0 and 2 received)
        assert extract_active_chunk_ids(first_merge) == {0, 2}
        # Template still has all chunks tagged
        assert "<chunk id=0>A</chunk>" in first_merge
        assert "<chunk id=1>B</chunk>" in first_merge
        assert "<chunk id=2>C</chunk>" in first_merge

        # Later, got chunk 1
        chunk1 = '<message uuid="abc" active="1"><chunk id=0>A</chunk><chunk id=1>B</chunk><chunk id=2>C</chunk></message>'

        # Merge the already-merged result with chunk 1
        result = merge_audio_script_gold([first_merge, chunk1])

        # Should have all chunks in active now
        assert extract_active_chunk_ids(result) == {0, 1, 2}


class TestExtractMessageUuid:
    """Tests for extract_message_uuid utility function."""

    def test_extract_uuid_from_valid_format(self):
        """Test extracting UUID from valid audio_script_gold format."""
        from tau2.agent.base.streaming_utils import extract_message_uuid

        script_gold = '<message uuid="abc-123-def"><chunk id=0>Hello</chunk></message>'
        result = extract_message_uuid(script_gold)
        assert result == "abc-123-def"

    def test_extract_uuid_returns_none_for_invalid_format(self):
        """Test that invalid format returns None."""
        from tau2.agent.base.streaming_utils import extract_message_uuid

        assert extract_message_uuid("plain text") is None
        assert extract_message_uuid("<message>no uuid</message>") is None

    def test_extract_uuid_returns_none_for_none_input(self):
        """Test that None input returns None."""
        from tau2.agent.base.streaming_utils import extract_message_uuid

        assert extract_message_uuid(None) is None

    def test_extract_uuid_returns_none_for_empty_string(self):
        """Test that empty string returns None."""
        from tau2.agent.base.streaming_utils import extract_message_uuid

        assert extract_message_uuid("") is None


class TestExtractActiveChunkIds:
    """Tests for extract_active_chunk_ids utility function."""

    def test_extract_single_active_id(self):
        """Test extracting a single active chunk ID."""
        from tau2.agent.base.streaming_utils import extract_active_chunk_ids

        script = '<message uuid="x" active="0"><chunk id=0>A</chunk><chunk id=1>B</chunk></message>'
        result = extract_active_chunk_ids(script)
        assert result == {0}

    def test_extract_multiple_active_ids(self):
        """Test extracting multiple active chunk IDs (merged result)."""
        from tau2.agent.base.streaming_utils import extract_active_chunk_ids

        script = '<message uuid="x" active="0,2,5"><chunk id=0>A</chunk><chunk id=1>B</chunk></message>'
        result = extract_active_chunk_ids(script)
        assert result == {0, 2, 5}

    def test_extract_no_active_attribute(self):
        """Test extracting when no active attribute present."""
        from tau2.agent.base.streaming_utils import extract_active_chunk_ids

        script = '<message uuid="x"><chunk id=0>A</chunk></message>'
        result = extract_active_chunk_ids(script)
        assert result == set()


class TestAudioChunkingAndMergingIntegration:
    """Integration tests for the full chunking and merging workflow."""

    @pytest.fixture
    def audio_chunker(self):
        """Create an audio chunker."""

        class SimpleAudioChunker(AudioChunkingMixin):
            def _next_turn_taking_action(self, state):
                return "generate_message"

            def _should_respond_to_chunk(self, incoming_chunk, state):
                return True

            def speech_detection(self, incoming_chunk):
                return True

            def _perform_turn_taking_action(self, state, action):
                return None, state

            def _process_tool_result(self, tool_result, state):
                return None, state

            def _emit_waiting_chunk(self, state):
                return None, state

        return SimpleAudioChunker(chunk_size=50)

    def test_chunk_then_merge_preserves_audio_script_gold_structure(
        self, audio_chunker
    ):
        """Test that chunking then merging all chunks preserves structure."""
        import re

        sample_rate = 16000
        num_samples = 150  # 3 chunks
        audio_bytes = b"\x00\x01" * num_samples

        audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
        )

        message = AssistantMessage(
            role="assistant",
            content="Hello world!",
            is_audio=True,
            audio_content=base64.b64encode(audio_bytes).decode("utf-8"),
            audio_format=audio_format,
            audio_script_gold="Hello world!",
        )

        # Chunk the message
        chunks = audio_chunker._create_chunk_messages(message)
        assert len(chunks) == 3

        # Merge all chunks
        merged = AssistantMessage.merge_chunks(chunks)

        # audio_script_gold should have all chunk tags (all received)
        assert merged.audio_script_gold is not None
        assert "<chunk id=0>" in merged.audio_script_gold
        assert "<chunk id=1>" in merged.audio_script_gold
        assert "<chunk id=2>" in merged.audio_script_gold

        # Text content should be preserved
        text_only = re.sub(r"<[^>]+>", "", merged.audio_script_gold)
        assert text_only == "Hello world!"

    def test_partial_chunks_merge_shows_missing(self, audio_chunker):
        """Test that merging partial chunks shows which ones are missing via active attr."""
        from tau2.agent.base.streaming_utils import extract_active_chunk_ids

        sample_rate = 16000
        num_samples = 150  # 3 chunks
        audio_bytes = b"\x00\x01" * num_samples

        audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
        )

        message = AssistantMessage(
            role="assistant",
            content="ABC",
            is_audio=True,
            audio_content=base64.b64encode(audio_bytes).decode("utf-8"),
            audio_format=audio_format,
            audio_script_gold="ABC",
        )

        # Chunk the message
        chunks = audio_chunker._create_chunk_messages(message)
        assert len(chunks) == 3

        # Only merge chunks 0 and 2 (skip chunk 1)
        partial_chunks = [chunks[0], chunks[2]]
        merged = AssistantMessage.merge_chunks(partial_chunks)

        # audio_script_gold should have active="0,2" to show which chunks were received
        assert merged.audio_script_gold is not None
        assert extract_active_chunk_ids(merged.audio_script_gold) == {0, 2}

        # All chunks are still in the template (for reference)
        assert "<chunk id=0>" in merged.audio_script_gold
        assert "<chunk id=1>" in merged.audio_script_gold
        assert "<chunk id=2>" in merged.audio_script_gold

        # Missing chunks can be identified by comparing all vs active chunk IDs
        from tau2.agent.base.streaming_utils import extract_all_chunk_ids

        missing = extract_all_chunk_ids(
            merged.audio_script_gold
        ) - extract_active_chunk_ids(merged.audio_script_gold)
        assert missing == {1}
