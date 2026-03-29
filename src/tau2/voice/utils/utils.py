from pathlib import Path

from loguru import logger

from tau2.voice_config import BACKGROUND_NOISE_CONTINUOUS_DIR, BURST_NOISE_DIR


def get_noise_files(noise_dir: Path) -> list[Path]:
    """Get all noise files in a directory"""
    return list(noise_dir.glob("*.wav"))


def get_burst_noise_files(burst_noise_dir: Path) -> list[Path]:
    """Get all burst noise files in a directory"""
    return list(burst_noise_dir.glob("*.wav"))


CONTINUOUS_NOISE_FILES = get_noise_files(BACKGROUND_NOISE_CONTINUOUS_DIR)
BURST_NOISE_FILES = get_burst_noise_files(BURST_NOISE_DIR)

if len(CONTINUOUS_NOISE_FILES) == 0:
    logger.warning(
        f"No continuous noise files found in the directory {BACKGROUND_NOISE_CONTINUOUS_DIR}"
    )
else:
    logger.info(
        f"Found {len(CONTINUOUS_NOISE_FILES)} continuous noise files in the directory {BACKGROUND_NOISE_CONTINUOUS_DIR}"
    )

if len(BURST_NOISE_FILES) == 0:
    logger.warning(f"No burst noise files found in the directory {BURST_NOISE_DIR}")
else:
    logger.info(
        f"Found {len(BURST_NOISE_FILES)} burst noise files in the directory {BURST_NOISE_DIR}"
    )
