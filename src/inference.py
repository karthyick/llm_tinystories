"""Inference module for TinyStories/WikiMini model with streaming support.

This module provides:
1. Model loading from checkpoint
2. Tokenizer loading
3. Streaming text generation for SSE
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Generator, Dict, Any, Tuple, TYPE_CHECKING
import logging
import threading

from src.model.transformer_block import WikiMiniModel
from src.data.tokenizer import load_tokenizer

if TYPE_CHECKING:
    from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles model and tokenizer loading with caching.

    Thread-safe singleton implementation using a lock for model loading.
    """

    _instance: Optional["ModelLoader"] = None
    _model: Optional[WikiMiniModel] = None
    _tokenizer: Optional["Tokenizer"] = None
    _device: Optional[torch.device] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def load(
        self,
        checkpoint_path: str = "checkpoints/checkpoint_best_loss.pth",
        tokenizer_path: str = "tokenizer/tinystories_10k",
        device: str = "cuda",
    ) -> Tuple[WikiMiniModel, "Tokenizer"]:
        """Load model and tokenizer if not already loaded.

        Thread-safe: Uses lock to prevent concurrent loading.

        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer directory
            device: Device to load model on ('cuda' or 'cpu')

        Returns:
            Tuple of (model, tokenizer)
        """
        if self._model is not None and self._tokenizer is not None:
            logger.info("Using cached model and tokenizer")
            return self._model, self._tokenizer

        with self._lock:
            # Double-check after acquiring lock
            if self._model is not None and self._tokenizer is not None:
                logger.info("Using cached model and tokenizer (after lock)")
                return self._model, self._tokenizer

            self._device = torch.device(device if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading model on device: {self._device}")

            # Load tokenizer
            self._tokenizer = load_tokenizer(tokenizer_path)
            logger.info(f"Tokenizer loaded: {len(self._tokenizer)} vocab size")

            # Load model
            self._model = self._load_model(checkpoint_path)
            self._model.eval()

            return self._model, self._tokenizer

    def _load_model(self, checkpoint_path: str) -> WikiMiniModel:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded WikiMiniModel
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self._device,
            weights_only=False
        )

        # Extract config
        if 'config' in checkpoint:
            config = checkpoint['config']['model']
        else:
            # Default config for TinyStories 24.5M
            config = {
                'vocab_size': 10000,
                'd_model': 512,
                'n_layers': 8,
                'n_heads': 8,
                'd_ffn': 1344,
                'max_seq_len': 2048,
                'dropout': 0.0,  # No dropout for inference
                'rope_percentage': 0.5,
                'rope_base': 10000,
                'rms_norm_eps': 1e-6,
                'tie_embeddings': True,
                'use_flash_attention': True,
            }

        # Override vocab size with tokenizer size
        config['vocab_size'] = len(self._tokenizer)
        config['dropout'] = 0.0  # Ensure no dropout during inference

        # Create model
        model = WikiMiniModel(config)

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self._device)

        # Log model info
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded: {params/1e6:.2f}M parameters on {self._device}")

        return model

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None and self._tokenizer is not None

    @property
    def model(self) -> Optional[WikiMiniModel]:
        """Get loaded model."""
        return self._model

    @property
    def tokenizer(self) -> Optional["Tokenizer"]:
        """Get loaded tokenizer."""
        return self._tokenizer

    @property
    def device(self) -> Optional[torch.device]:
        """Get device model is loaded on."""
        return self._device


class StreamingGenerator:
    """Streaming text generator with SSE support."""

    def __init__(self, model_loader: ModelLoader):
        """Initialize generator with model loader.

        Args:
            model_loader: ModelLoader instance with loaded model
        """
        self.model_loader = model_loader

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> Generator[str, None, None]:
        """Generate text with streaming output.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep top-k tokens for sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens

        Yields:
            Generated tokens one at a time
        """
        model = self.model_loader.model
        tokenizer = self.model_loader.tokenizer
        device = self.model_loader.device

        if model is None or tokenizer is None:
            raise RuntimeError("Model not loaded. Call model_loader.load() first.")

        # Tokenize prompt
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)],
            dtype=torch.long,
            device=device
        )

        # Track generated tokens for repetition penalty
        generated_ids = input_ids[0].tolist()

        # Yield prompt tokens first (optional - can be removed if not wanted)
        # For now, we only yield new tokens

        # Generate tokens
        for _ in range(max_tokens):
            # Get model predictions with mixed precision
            with torch.amp.autocast(
                device_type='cuda',
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_available()
            ):
                outputs = model(input_ids)

            logits = outputs['logits'][0, -1, :]  # Get last token logits

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids):
                    logits[token_id] /= repetition_penalty

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -float('inf')

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Decode and yield the new token
            new_token_text = tokenizer.decode([next_token.item()])
            yield new_token_text

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            generated_ids.append(next_token.item())

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate complete text (non-streaming).

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Keep top-k tokens for sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens

        Returns:
            Complete generated text
        """
        full_text = ""
        for token in self.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            full_text += token
        return full_text


# Global instances for singleton pattern
_model_loader: Optional[ModelLoader] = None
_generator: Optional[StreamingGenerator] = None


def get_model_loader() -> ModelLoader:
    """Get or create the global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def get_generator() -> StreamingGenerator:
    """Get or create the global generator instance."""
    global _generator
    if _generator is None:
        _generator = StreamingGenerator(get_model_loader())
    return _generator


def load_model(
    checkpoint_path: str = "checkpoints/checkpoint_best_loss.pth",
    tokenizer_path: str = "tokenizer/tinystories_10k",
    device: str = "cuda",
) -> Tuple[WikiMiniModel, "Tokenizer"]:
    """Convenience function to load model and tokenizer.

    Args:
        checkpoint_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer directory
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    loader = get_model_loader()
    return loader.load(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        device=device,
    )


if __name__ == "__main__":
    # Test the inference module
    import sys

    logging.basicConfig(level=logging.INFO)

    print("Loading model...")
    model, tokenizer = load_model()

    print(f"\nModel loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Test streaming generation
    print("\n" + "="*60)
    print("Testing streaming generation...")
    print("="*60)

    generator = get_generator()
    prompt = "Once upon a time"

    print(f"\nPrompt: {prompt}")
    print("Generated: ", end="", flush=True)

    for token in generator.generate_stream(prompt, max_tokens=50):
        print(token, end="", flush=True)

    print("\n\n" + "="*60)
    print("Test complete!")
    print("="*60)
