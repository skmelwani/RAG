"""Project settings, read from environment variables / a .env file.

Every tunable value lives here, so experiments (e.g. chunk size 256 vs 512)
change a setting, never the code. Defaults are sensible for a laptop run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# src/config.py -> project root is one level above src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Reads .env into os.environ. Real environment variables win over .env,
# which is what you want in Docker/CI (override without editing files).
load_dotenv(PROJECT_ROOT / ".env")


def _path(name: str, default: str) -> Path:
    """Read a path setting. Relative paths are resolved from the project root,
    so scripts work no matter which folder you run them from."""
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError as e:
        # Fail at startup with a clear message, not deep inside the chunker.
        raise ValueError(f"Setting {name} must be an integer, got {value!r}") from e


@dataclass(frozen=True)
class Settings:
    raw_dir: Path
    processed_dir: Path
    manifest_path: Path
    chroma_dir: Path  # Chroma's persistent database folder
    embedding_model: str  # also its tokenizer measures chunk size
    embedding_device: str  # "cpu", "mps" (Apple GPU) or "cuda"
    embedding_batch_size: int
    query_prefix: str  # instruction bge expects before queries (not documents)
    chunk_size: int  # max tokens per chunk
    chunk_overlap: int  # tokens shared between neighbouring chunks
    top_k: int  # how many chunks a search returns
    log_level: str


def get_settings() -> Settings:
    """Build settings from the environment. Call this once at startup."""
    return Settings(
        raw_dir=_path("RAW_DIR", "data/raw"),
        processed_dir=_path("PROCESSED_DIR", "data/processed"),
        manifest_path=_path("MANIFEST_PATH", "data/manifest.csv"),
        chroma_dir=_path("CHROMA_DIR", "data/chroma"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        # CPU by default: PyTorch's Apple GPU backend crashes on macOS 14.
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", 32),
        query_prefix=os.getenv(
            "QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
        ),
        chunk_size=_int("CHUNK_SIZE", 256),
        chunk_overlap=_int("CHUNK_OVERLAP", 40),
        top_k=_int("TOP_K", 5),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
