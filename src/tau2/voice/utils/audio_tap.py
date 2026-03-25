"""Audio tap for recording audio at pipeline stages.

Accumulates audio chunks in memory and writes a WAV file on save().
Used for diagnosing and comparing audio signal properties across
different pipeline paths (e.g., human user vs. user simulator).
"""

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.message import ParticipantMessageBase
from tau2.voice.utils.audio_io import save_wav_file


class AudioTap:
    """Records audio at a single pipeline stage.

    Accumulates raw audio bytes from each chunk, then writes a single
    WAV file when save() is called. Companded formats (ulaw/alaw) are
    automatically decoded to PCM16 for the WAV output.

    Supports recording from raw bytes, messages, or numpy arrays
    (for multitrack pipeline stages).
    """

    def __init__(
        self,
        name: str,
        output_dir: Path,
        sample_rate: int,
        encoding: AudioEncoding,
    ):
        self.name = name
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.encoding = encoding
        self._chunks: list[bytes] = []
        self._total_bytes = 0

    def record(self, audio_bytes: bytes) -> None:
        """Record raw audio bytes from one chunk."""
        if audio_bytes:
            self._chunks.append(audio_bytes)
            self._total_bytes += len(audio_bytes)

    def record_numpy(self, samples: np.ndarray) -> None:
        """Record a numpy array as one chunk, converting to int16 for WAV storage.

        Accepts any numeric dtype (float64 tracks are clipped and quantized here).
        """
        i16 = np.clip(samples, -32768, 32767).astype(np.int16)
        audio_bytes = i16.tobytes()
        if audio_bytes:
            self._chunks.append(audio_bytes)
            self._total_bytes += len(audio_bytes)

    def record_message(self, msg: Optional[ParticipantMessageBase]) -> None:
        """Record audio from a message (extracts bytes automatically).

        Works with any ParticipantMessageBase subclass (UserMessage,
        AssistantMessage) that carries audio_content.
        """
        if msg is None or not msg.audio_content:
            return
        audio_bytes = msg.get_audio_bytes()
        if audio_bytes:
            self.record(audio_bytes)

    def save(self) -> Optional[Path]:
        """Concatenate all chunks and write a WAV file.

        Returns the path to the saved file, or None if no audio was recorded.
        """
        if not self._chunks:
            logger.debug(f"AudioTap '{self.name}': no audio recorded, skipping save")
            return None

        all_bytes = b"".join(self._chunks)
        audio = AudioData(
            data=all_bytes,
            format=AudioFormat(
                encoding=self.encoding,
                sample_rate=self.sample_rate,
                channels=1,
            ),
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.name}.wav"

        if path.exists():
            path.unlink()
        save_wav_file(audio, path)
        logger.info(
            f"AudioTap '{self.name}': saved {len(all_bytes)} bytes "
            f"({len(self._chunks)} chunks) to {path}"
        )
        return path

    def __repr__(self) -> str:
        return (
            f"AudioTap(name={self.name!r}, chunks={len(self._chunks)}, "
            f"bytes={self._total_bytes}, encoding={self.encoding})"
        )
