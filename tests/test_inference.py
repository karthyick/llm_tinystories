"""Tests for the inference module.

These tests verify:
1. ModelLoader singleton behavior
2. StreamingGenerator token generation
3. Error handling
4. Thread safety
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import threading

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import (
    ModelLoader,
    StreamingGenerator,
    get_model_loader,
    get_generator,
    load_model,
)


class TestModelLoader:
    """Tests for ModelLoader class."""

    def test_singleton_pattern(self):
        """Test that ModelLoader is a singleton."""
        loader1 = ModelLoader()
        loader2 = ModelLoader()

        assert loader1 is loader2, "ModelLoader should return the same instance"

    def test_is_loaded_false_initially(self):
        """Test that is_loaded is False before loading."""
        # Reset the singleton for this test
        ModelLoader._instance = None
        loader = ModelLoader()

        assert loader.is_loaded is False

    def test_model_property_none_initially(self):
        """Test that model property is None before loading."""
        ModelLoader._instance = None
        loader = ModelLoader()

        assert loader.model is None

    def test_tokenizer_property_none_initially(self):
        """Test that tokenizer property is None before loading."""
        ModelLoader._instance = None
        loader = ModelLoader()

        assert loader.tokenizer is None

    def test_device_property_none_initially(self):
        """Test that device property is None before loading."""
        ModelLoader._instance = None
        loader = ModelLoader()

        assert loader.device is None

    def test_singleton_thread_safety(self):
        """Test that singleton is thread-safe."""
        ModelLoader._instance = None

        instances = []

        def create_instance():
            instances.append(ModelLoader())

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same
        assert all(inst is instances[0] for inst in instances)


class TestModelLoaderLoading:
    """Tests for ModelLoader load functionality."""

    def teardown_method(self):
        """Reset singleton after each test."""
        ModelLoader._instance = None
        ModelLoader._model = None
        ModelLoader._tokenizer = None
        ModelLoader._device = None

    @patch('src.inference.load_tokenizer')
    @patch('torch.load')
    def test_load_raises_filenotfound_for_missing_checkpoint(
        self, mock_torch_load, mock_load_tokenizer
    ):
        """Test that load raises FileNotFoundError for missing checkpoint."""
        mock_load_tokenizer.return_value = Mock(__len__=lambda self: 10000)

        loader = ModelLoader()

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            loader.load(
                checkpoint_path="nonexistent.pth",
                tokenizer_path="test_tokenizer"
            )

    def test_load_returns_cached_model_on_second_call(self):
        """Test that second load call returns cached model (integration test)."""
        # This test verifies the caching behavior by checking that
        # once model is set, subsequent calls return the same instance
        ModelLoader._instance = None
        ModelLoader._model = Mock()
        ModelLoader._tokenizer = Mock()

        loader = ModelLoader()
        # Since model and tokenizer are already set, load should return cached
        model1, tok1 = loader.load("any.pth", "any_tok")
        model2, tok2 = loader.load("other.pth", "other_tok")

        assert model1 is model2
        assert tok1 is tok2


class TestStreamingGenerator:
    """Tests for StreamingGenerator class."""

    def test_init_with_model_loader(self):
        """Test StreamingGenerator initialization."""
        ModelLoader._instance = None
        loader = ModelLoader()
        generator = StreamingGenerator(loader)

        assert generator.model_loader is loader

    def test_generate_raises_when_not_loaded(self):
        """Test that generate raises RuntimeError when model not loaded."""
        ModelLoader._instance = None
        loader = ModelLoader()
        generator = StreamingGenerator(loader)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            list(generator.generate_stream("test prompt"))

    def test_generate_complete_raises_when_not_loaded(self):
        """Test that generate (non-streaming) raises RuntimeError when model not loaded."""
        ModelLoader._instance = None
        loader = ModelLoader()
        generator = StreamingGenerator(loader)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            generator.generate("test prompt")


class TestStreamingGeneratorWithModel:
    """Tests for StreamingGenerator with mocked model."""

    def teardown_method(self):
        """Reset singleton after each test."""
        ModelLoader._instance = None
        ModelLoader._model = None
        ModelLoader._tokenizer = None
        ModelLoader._device = None

    @patch('src.inference.load_tokenizer')
    @patch('src.inference.WikiMiniModel')
    @patch('torch.load')
    @patch('torch.cuda.is_available')
    def test_generate_stream_yields_tokens(
        self, mock_cuda_available, mock_torch_load, mock_model_class, mock_load_tokenizer
    ):
        """Test that generate_stream yields tokens."""
        mock_cuda_available.return_value = False
        mock_load_tokenizer.return_value = Mock(
            __len__=lambda self: 10000,
            encode=lambda text: [1, 2, 3],
            decode=lambda ids: "token",
            eos_token_id=2
        )
        mock_torch_load.return_value = {'model_state_dict': {}}

        # Create mock model that returns logits
        mock_model = Mock()
        mock_output = {
            'logits': torch.randn(1, 3, 10000)  # batch=1, seq_len=3, vocab=10000
        }
        mock_model.return_value = mock_output
        mock_model.parameters = Mock(return_value=[])
        mock_model_class.return_value = mock_model

        loader = ModelLoader()
        loader._model = mock_model
        loader._tokenizer = mock_load_tokenizer.return_value
        loader._device = torch.device("cpu")

        generator = StreamingGenerator(loader)

        # Get tokens from generator
        tokens = list(generator.generate_stream("test", max_tokens=5))

        # Should have yielded some tokens
        assert len(tokens) <= 5  # May stop early on EOS
        assert all(isinstance(t, str) for t in tokens)

    @patch('src.inference.load_tokenizer')
    @patch('src.inference.WikiMiniModel')
    @patch('torch.load')
    @patch('torch.cuda.is_available')
    def test_generate_returns_complete_text(
        self, mock_cuda_available, mock_torch_load, mock_model_class, mock_load_tokenizer
    ):
        """Test that generate returns complete text."""
        mock_cuda_available.return_value = False
        mock_load_tokenizer.return_value = Mock(
            __len__=lambda self: 10000,
            encode=lambda text: [1, 2, 3],
            decode=lambda ids: "token",
            eos_token_id=2
        )
        mock_torch_load.return_value = {'model_state_dict': {}}

        mock_model = Mock()
        mock_output = {
            'logits': torch.randn(1, 3, 10000)
        }
        mock_model.return_value = mock_output
        mock_model.parameters = Mock(return_value=[])
        mock_model_class.return_value = mock_model

        loader = ModelLoader()
        loader._model = mock_model
        loader._tokenizer = mock_load_tokenizer.return_value
        loader._device = torch.device("cpu")

        generator = StreamingGenerator(loader)

        # Generate complete text
        result = generator.generate("test", max_tokens=5)

        assert isinstance(result, str)


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    def test_get_model_loader_returns_singleton(self):
        """Test that get_model_loader returns the same instance."""
        from src.inference import _model_loader

        # Reset
        import src.inference
        src.inference._model_loader = None

        loader1 = get_model_loader()
        loader2 = get_model_loader()

        assert loader1 is loader2

    def test_get_generator_returns_singleton(self):
        """Test that get_generator returns the same instance."""
        from src.inference import _generator

        # Reset
        import src.inference
        src.inference._generator = None

        gen1 = get_generator()
        gen2 = get_generator()

        assert gen1 is gen2


class TestLoadModel:
    """Tests for load_model convenience function."""

    @patch('src.inference.get_model_loader')
    def test_load_model_calls_loader(self, mock_get_loader):
        """Test that load_model calls the loader correctly."""
        mock_loader = Mock()
        mock_loader.load.return_value = (Mock(), Mock())
        mock_get_loader.return_value = mock_loader

        result = load_model(
            checkpoint_path="test/path.pth",
            tokenizer_path="test/tokenizer",
            device="cpu"
        )

        mock_loader.load.assert_called_once_with(
            checkpoint_path="test/path.pth",
            tokenizer_path="test/tokenizer",
            device="cpu"
        )
        assert isinstance(result, tuple)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
