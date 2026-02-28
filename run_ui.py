#!/usr/bin/env python3
"""Startup script for TinyStories Web UI.

This script starts the FastAPI server on port 7779 with the TinyStories model.
The model is loaded on startup and the UI is available at http://localhost:7779

Usage:
    python run_ui.py

Environment Variables:
    CHECKPOINT_PATH: Path to model checkpoint (default: checkpoints/checkpoint_best_loss.pth)
    TOKENIZER_PATH: Path to tokenizer directory (default: tokenizer/tinystories_10k)
    DEVICE: Device to run inference on (default: cuda)
    PORT: Server port (default: 7779)
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Start the TinyStories Web UI server."""
    # Get configuration from environment or use defaults
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 7779))

    # Log startup information
    logger.info("=" * 60)
    logger.info("Starting TinyStories Web UI")
    logger.info("=" * 60)
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Checkpoint: {os.environ.get('CHECKPOINT_PATH', 'checkpoints/checkpoint_best_loss.pth')}")
    logger.info(f"Tokenizer: {os.environ.get('TOKENIZER_PATH', 'tokenizer/tinystories_10k')}")
    logger.info(f"Device: {os.environ.get('DEVICE', 'cuda')}")
    logger.info("=" * 60)
    logger.info(f"Open http://localhost:{port} in your browser")
    logger.info("=" * 60)

    # Run the server
    uvicorn.run(
        "src.server:app",
        host=host,
        port=port,
        reload=False,  # Disable reload to prevent model from loading twice
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
