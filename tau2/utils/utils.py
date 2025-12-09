import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from deepdiff import DeepDiff
from dotenv import load_dotenv
from loguru import logger

res = load_dotenv()
# if not res:
#     logger.warning("No .env file found")

# SOURCE_DIR = Path('/lustre/fsw/portfolios/nvr/users/hongjins/data/tool_use/original/tau2')  # .parents[2]
# DATA_DIR = SOURCE_DIR / "data"
DATA_DIR = Path('/lustre/fsw/portfolios/nvr/users/hongjins/data/tool_use/original')


def get_dict_hash(obj: dict) -> str:
    """
    Generate a unique hash for dict.
    Returns a hex string representation of the hash.
    """
    hash_string = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(hash_string.encode()).hexdigest()


def show_dict_diff(dict1: dict, dict2: dict) -> str:
    """
    Show the difference between two dictionaries.
    """
    diff = DeepDiff(dict1, dict2)
    return diff


def get_now() -> str:
    """
    Returns the current date and time in the format YYYYMMDD_HHMMSS.
    """
    now = datetime.now()
    return format_time(now)


def format_time(time: datetime) -> str:
    """
    Format the time in the format YYYYMMDD_HHMMSS.
    """
    return time.isoformat()


def get_commit_hash() -> str:
    """
    Get the commit hash of the current directory.
    """
    raise ValueError('debug')
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
            .split("\n")[0]
        )
    except Exception as e:
        logger.error(f"Failed to get git hash: {e}")
        commit_hash = "unknown"
    return commit_hash
