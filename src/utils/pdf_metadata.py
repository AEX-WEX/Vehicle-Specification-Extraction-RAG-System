"""PDF metadata tracking utilities."""

import logging
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def get_pdf_hash(pdf_path: str) -> str:
    """
    Calculate MD5 hash of PDF file.

    Args:
        pdf_path: Path to PDF file

    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_indexed_pdf_metadata(index_dir: str) -> Optional[Dict]:
    """
    Load metadata about the previously indexed PDF.

    Args:
        index_dir: Path to index directory

    Returns:
        Metadata dict or None if metadata file doesn't exist
    """
    metadata_path = Path(index_dir) / "indexed_pdf_metadata.json"

    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning(f"Invalid metadata file at {metadata_path}")
        return None


def save_indexed_pdf_metadata(
    index_dir: str,
    pdf_path: str,
    num_chunks: int,
    embedding_model: str,
    index_type: str
) -> None:
    """
    Save metadata about the indexed PDF.

    Args:
        index_dir: Path to index directory
        pdf_path: Path to the indexed PDF
        num_chunks: Number of chunks created
        embedding_model: Name of embedding model used
        index_type: Type of FAISS index used
    """
    metadata = {
        "pdf_filename": Path(pdf_path).name,
        "pdf_path": str(pdf_path),
        "pdf_hash": get_pdf_hash(pdf_path),
        "indexed_at": datetime.now().isoformat(),
        "num_chunks": num_chunks,
        "embedding_model": embedding_model,
        "index_type": index_type
    }

    os.makedirs(index_dir, exist_ok=True)
    metadata_path = Path(index_dir) / "indexed_pdf_metadata.json"

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata: {metadata_path}")


def is_pdf_different(pdf_path: str, index_dir: str) -> bool:
    """
    Check if the provided PDF is different from the previously indexed PDF.

    Args:
        pdf_path: Path to current PDF
        index_dir: Path to index directory

    Returns:
        True if PDF is different (or no previous PDF), False if same
    """
    old_metadata = load_indexed_pdf_metadata(index_dir)

    if old_metadata is None:
        logger.info("No previous index metadata found - will build new index")
        return True

    current_hash = get_pdf_hash(pdf_path)
    previous_hash = old_metadata.get("pdf_hash")

    if current_hash != previous_hash:
        logger.info(f"PDF hash changed - detected new/different PDF")
        logger.debug(f"Old hash: {previous_hash}, New hash: {current_hash}")
        return True

    logger.info("PDF hash matches previous index - using existing index")
    return False
