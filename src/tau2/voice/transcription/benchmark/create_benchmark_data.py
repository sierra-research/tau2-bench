"""
Create benchmark data for voice transcription evaluation.

This script extracts transcription data from voice simulations and creates
structured CSV files for further analysis.

Usage:
    python -m tau2.voice.transcription.benchmark.create_benchmark_data [options]
"""

import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_turn_uuid_from_path(audio_path: str) -> Optional[str]:
    """
    Extract turn UUID from audio path.

    Args:
        audio_path: Path to audio file containing turn UUID

    Returns:
        Turn UUID or None if not found
    """
    try:
        # Audio path format: .../voice/sim_<sim_uuid>/turn_<turn_uuid>/speech.wav
        path_parts = Path(audio_path).parts
        for part in path_parts:
            if part.startswith("turn_"):
                return part.replace("turn_", "")
        return None
    except Exception as e:
        logger.warning(f"Error extracting turn UUID from {audio_path}: {e}")
        return None


def load_voice_simulations(
    simulations_dir: Path, limit: Optional[int] = None
) -> List[Tuple[Path, dict]]:
    """
    Find and load all voice simulation results.

    Args:
        simulations_dir: Directory containing simulation results
        limit: Maximum number of simulations to load

    Returns:
        List of (path, data) tuples for voice simulations
    """
    voice_simulations = []
    simulation_dirs = sorted([d for d in simulations_dir.iterdir() if d.is_dir()])

    for sim_dir in simulation_dirs:
        results_path = sim_dir / "results.json"

        try:
            with open(results_path, "r") as f:
                data = json.load(f)

            # Check if this is a voice simulation
            if data.get("info", {}).get("voice_info") is not None:
                voice_simulations.append((sim_dir, data))
                logger.info(f"Found voice simulation: {sim_dir.name}")

                if limit and len(voice_simulations) >= limit:
                    break
        except Exception as e:
            logger.warning(f"Error loading {results_path}: {e}")

    logger.info(f"Found {len(voice_simulations)} voice simulations")
    return voice_simulations


def extract_transcription_pairs(simulation_data: dict) -> List[Dict]:
    """
    Extract gold/LLM transcript pairs from simulation.

    Args:
        simulation_data: Loaded simulation results

    Returns:
        List of transcription pair dictionaries
    """
    pairs = []

    # Extract voice info
    voice_info = simulation_data["info"]["voice_info"]
    model = voice_info.get("transcription_model", "")
    transcription_config = voice_info.get("transcription_config", {})
    locale = (
        transcription_config.get("language") or "en-US"
    )  # Default to en-US if empty

    # Remove language from other_config to avoid duplication
    other_config = {k: v for k, v in transcription_config.items() if k != "language"}
    other_config_str = json.dumps(other_config) if other_config else ""

    # Process each simulation run
    for sim in simulation_data.get("simulations", []):
        simulation_id = sim["id"]

        for message in sim.get("messages", []):
            if (
                message.get("role") == "user"
                and message.get("audio_path")
                and message.get("content") is not None
            ):
                turn_id = extract_turn_uuid_from_path(message["audio_path"])
                if not turn_id:
                    logger.warning(
                        f"Could not extract turn ID from {message.get('audio_path')}"
                    )
                    continue

                gold_transcript = message["content"]
                if message.get("raw_data") and message["raw_data"].get("message"):
                    llm_transcript = message["raw_data"]["message"].get("content", "")
                else:
                    llm_transcript = ""

                # Skip if either transcript is missing
                if not gold_transcript and not llm_transcript:
                    continue

                pairs.append(
                    {
                        "simulation_id": simulation_id,
                        "turn_id": turn_id,
                        "model": model,
                        "locale": locale,
                        "other_config": other_config_str,
                        "gold_transcript": gold_transcript,
                        "llm_transcript": llm_transcript,
                    }
                )

    logger.info(f"Extracted {len(pairs)} transcription pairs")
    if len(pairs) == 0:
        # Check what data is available for debugging
        num_simulations = len(simulation_data.get("simulations", []))
        logger.debug(f"Found {num_simulations} simulation runs in data")
        if num_simulations > 0:
            first_sim = simulation_data["simulations"][0]
            num_messages = len(first_sim.get("messages", []))
            logger.debug(f"First simulation has {num_messages} messages")

    return pairs


def save_benchmark_data(
    transcription_pairs: List[Dict],
    simulation_data: dict,
    output_dir: Path,
    simulation_name: str,
) -> None:
    """
    Save CSV and params.json for a simulation.

    Args:
        transcription_pairs: List of transcription data
        simulation_data: Original simulation data
        output_dir: Directory to save benchmark data
        simulation_name: Name of the simulation
    """
    if not transcription_pairs:
        logger.warning(f"No transcription pairs found for {simulation_name}")
        return

    # Create unique directory for this simulation
    benchmark_id = str(uuid.uuid4())
    sim_output_dir = output_dir / benchmark_id
    sim_output_dir.mkdir(parents=True, exist_ok=True)

    # Save transcription data as CSV
    df = pd.DataFrame(transcription_pairs)
    csv_path = sim_output_dir / "transcription_data.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} rows to {csv_path}")

    # Save metadata
    params = {
        "benchmark_id": benchmark_id,
        "simulation_name": simulation_name,
        "simulation_info": simulation_data["info"],
        "num_transcription_pairs": len(transcription_pairs),
    }

    params_path = sim_output_dir / "params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved parameters to {params_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract transcription benchmark data from voice simulations"
    )
    parser.add_argument(
        "--simulations-dir",
        type=Path,
        default=Path("data/simulations"),
        help="Directory containing simulation results (default: data/simulations)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmark"),
        help="Directory to save benchmark data (default: data/benchmark)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of simulations to process",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load voice simulations
    simulations = load_voice_simulations(args.simulations_dir, args.limit)

    if not simulations:
        logger.warning("No voice simulations found")
        return

    # Process each simulation
    for sim_dir, sim_data in simulations:
        logger.info(f"Processing {sim_dir.name}...")

        try:
            # Extract transcription pairs
            pairs = extract_transcription_pairs(sim_data)

            # Save benchmark data
            save_benchmark_data(pairs, sim_data, args.output_dir, sim_dir.name)
        except Exception as e:
            logger.error(f"Error processing {sim_dir.name}: {e}")
            continue

    logger.info("Benchmark data extraction complete")


if __name__ == "__main__":
    main()
