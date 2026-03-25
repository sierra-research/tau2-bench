"""Audio I/O utilities for loading and saving audio files."""

import wave
from pathlib import Path
from typing import Literal

import pyaudio
from loguru import logger

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.voice.utils.audio_preprocessing import convert_to_pcm16

# Mapping from WAV sample width to AudioEncoding for uncompressed PCM
_WAV_SAMPWIDTH_TO_ENCODING: dict[int, AudioEncoding] = {
    1: AudioEncoding.PCM_U8,  # 8-bit WAV is unsigned
    2: AudioEncoding.PCM_S16LE,
    3: AudioEncoding.PCM_S24LE,
    4: AudioEncoding.PCM_S32LE,
}


def load_wav_file(path: str | Path) -> AudioData:
    """
    Load a WAV file into an AudioData object.
    Supports PCM (8/16/24/32-bit), μ-law, and A-law WAV files.
    """
    with wave.open(str(path), "rb") as wav:
        channels: Literal[1, 2] = wav.getnchannels()  # type: ignore[assignment]
        sample_rate = wav.getframerate()
        sampwidth = wav.getsampwidth()
        nframes = wav.getnframes()
        comptype = wav.getcomptype()  # 'NONE', 'ULAW', 'ALAW', etc.
        compname = wav.getcompname()

        raw_data = wav.readframes(nframes)

        # Determine encoding based on compression type
        if comptype == "NONE":
            # Uncompressed PCM - map sample width to encoding
            encoding = _WAV_SAMPWIDTH_TO_ENCODING.get(sampwidth)
            if encoding is None:
                raise ValueError(f"Unsupported PCM sample width: {sampwidth}")
        elif comptype.upper() == "ULAW":
            encoding = AudioEncoding.ULAW
        elif comptype.upper() == "ALAW":
            encoding = AudioEncoding.ALAW
        else:
            raise ValueError(f"Unsupported WAV encoding: {compname} ({comptype})")

        fmt = AudioFormat(
            encoding=encoding,
            sample_rate=sample_rate,
            channels=channels,
        )
        return AudioData(data=raw_data, format=fmt, audio_path=path)


def save_wav_file(audio: AudioData, output_path: str | Path) -> None:
    """
    Save AudioData as a WAV file.

    Companded formats (ULAW/ALAW) are converted to PCM_S16LE before saving.

    Args:
        audio: AudioData object containing bytes and format metadata
        output_path: Path to save the WAV file
    """
    # Convert companded formats to PCM for WAV output
    if audio.format.encoding.is_companded:
        logger.warning(
            f"Converting {audio.format.encoding} to PCM_S16LE for WAV file..."
        )
        audio = convert_to_pcm16(audio)

    if isinstance(output_path, str):
        output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix != ".wav":
        raise ValueError(f"Output path must be a WAV file, got {output_path.suffix}")

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(audio.format.channels)
        wav_file.setsampwidth(audio.format.sample_width)
        wav_file.setframerate(audio.format.sample_rate)
        wav_file.setcomptype("NONE", "not compressed")  # Always save as linear PCM
        wav_file.writeframes(audio.data)


def play_audio(audio_data: AudioData) -> None:
    """Play AudioData using pyaudio. Converts to PCM_S16LE before playback."""
    p = pyaudio.PyAudio()

    # Convert to PCM_S16LE for playback
    audio_data = convert_to_pcm16(audio_data)

    stream = p.open(
        format=pyaudio.paInt16,  # Always 16-bit after convert_to_pcm16
        channels=audio_data.format.channels,
        rate=audio_data.format.sample_rate,
        output=True,
    )

    try:
        stream.write(audio_data.data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
