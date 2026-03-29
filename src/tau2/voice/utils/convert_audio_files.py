"""
Batch audio file conversion utility.

Converts audio files to a standardized format (PCM 16-bit mono at a target sample rate).

Usage:
    python -m tau2.voice.utils.convert_audio_files <input_dir> <output_dir> [--sample-rate 16000]

Example:
    python -m tau2.voice.utils.convert_audio_files \
        data/voice/background_noise_audio_pcm_mono_verified/raw/urbansound8k/audio \
        data/voice/background_noise_audio_pcm_mono_verified/raw/urbansound8k_converted \
        --sample-rate 16000
"""

import argparse
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from tau2.config import DEFAULT_PCM_SAMPLE_RATE
from tau2.voice.utils.audio_io import load_wav_file, save_wav_file
from tau2.voice.utils.audio_preprocessing import (
    convert_to_mono,
    convert_to_pcm16,
    resample_audio,
)


def convert_audio_file(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int = DEFAULT_PCM_SAMPLE_RATE,
) -> bool:
    """
    Convert a single audio file to PCM 16-bit mono at the target sample rate.

    Args:
        input_path: Path to input WAV file
        output_path: Path to save converted WAV file
        target_sample_rate: Target sample rate in Hz (default: 16000)

    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        # Load the audio file
        audio = load_wav_file(input_path)

        # Convert to PCM 16-bit
        audio = convert_to_pcm16(audio)

        # Convert to mono
        audio = convert_to_mono(audio)

        # Resample to target rate
        audio = resample_audio(audio, target_sample_rate)

        # Save the converted file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()  # Remove existing file to allow overwrite
        save_wav_file(audio, output_path)

        return True

    except Exception as e:
        logger.error(f"Failed to convert {input_path}: {e}")
        return False


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    target_sample_rate: int = DEFAULT_PCM_SAMPLE_RATE,
    file_pattern: str = "*.wav",
    preserve_structure: bool = True,
) -> tuple[int, int]:
    """
    Convert all audio files in a directory to the target format.

    Args:
        input_dir: Input directory containing WAV files
        output_dir: Output directory for converted files
        target_sample_rate: Target sample rate in Hz (default: 16000)
        file_pattern: Glob pattern for finding files (default: "*.wav")
        preserve_structure: If True, preserve subdirectory structure (default: True)

    Returns:
        Tuple of (success_count, failure_count)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Find all matching files
    input_files = list(input_dir.rglob(file_pattern))
    logger.info(
        f"Found {len(input_files)} files matching '{file_pattern}' in {input_dir}"
    )

    if not input_files:
        logger.warning("No files found to convert")
        return 0, 0

    success_count = 0
    failure_count = 0

    for input_path in tqdm(input_files, desc="Converting"):
        # Determine output path
        if preserve_structure:
            relative_path = input_path.relative_to(input_dir)
            output_path = output_dir / relative_path
        else:
            output_path = output_dir / input_path.name

        # Convert the file
        if convert_audio_file(input_path, output_path, target_sample_rate):
            success_count += 1
        else:
            failure_count += 1

    logger.info(
        f"Conversion complete: {success_count} succeeded, {failure_count} failed"
    )
    return success_count, failure_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio files to PCM 16-bit mono at a target sample rate."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing WAV files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_PCM_SAMPLE_RATE,
        help=f"Target sample rate in Hz (default: {DEFAULT_PCM_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.wav",
        help="Glob pattern for finding files (default: *.wav)",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Output all files to a flat directory (don't preserve structure)",
    )

    args = parser.parse_args()

    logger.info(f"Converting files from {args.input_dir} to {args.output_dir}")
    logger.info(f"Target format: PCM 16-bit mono @ {args.sample_rate} Hz")

    success, failure = convert_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_sample_rate=args.sample_rate,
        file_pattern=args.pattern,
        preserve_structure=not args.flat,
    )

    if failure > 0:
        exit(1)


if __name__ == "__main__":
    main()
