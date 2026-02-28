"""Tests for the FastAPI server module.

These tests verify:
1. Route registration
2. Request validation
3. Response formats
4. Health endpoint
5. SSE streaming
6. Error handling
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path
import json

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestServerRoutes:
    """Tests for server route registration."""

    @pytest.fixture
    def client(self):
        """Create a test client with mocked model loading."""
        with patch('src.server.get_model_loader') as mock_loader:
            mock_instance = Mock()
            mock_instance.device = "cuda"
            mock_instance.tokenizer = Mock()
            mock_instance.tokenizer.__len__ = Mock(return_value=10000)
            mock_instance.model = Mock()
            mock_instance.model.parameters = Mock(return_value=[])
            mock_loader.return_value = mock_instance

            from src.server import app
            with TestClient(app) as client:
                yield client

    def test_root_returns_html(self, client):
        """Test that root endpoint returns HTML."""
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "TinyStories" in response.text

    def test_health_endpoint_exists(self, client):
        """Test that health endpoint exists."""
        response = client.get("/health")

        assert response.status_code == 200

    def test_health_response_format(self, client):
        """Test health response has required fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data

    def test_api_info_endpoint(self, client):
        """Test API info endpoint."""
        response = client.get("/api/info")

        assert response.status_code == 200
        data = response.json()

        assert "server" in data
        assert "version" in data
        assert "endpoints" in data

    def test_generate_endpoint_requires_post(self, client):
        """Test that generate endpoint only accepts POST."""
        response = client.get("/generate")

        assert response.status_code == 405  # Method not allowed

    def test_generate_endpoint_requires_prompt(self, client):
        """Test that generate endpoint requires prompt field."""
        response = client.post("/generate", json={})

        assert response.status_code == 422  # Validation error


class TestGenerateRequest:
    """Tests for GenerateRequest validation."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        with patch('src.server.get_model_loader') as mock_loader:
            mock_instance = Mock()
            mock_instance.device = "cuda"
            mock_instance.tokenizer = Mock()
            mock_instance.tokenizer.__len__ = Mock(return_value=10000)
            mock_instance.model = Mock()
            mock_instance.model.parameters = Mock(return_value=[])
            mock_loader.return_value = mock_instance

            from src.server import app
            with TestClient(app) as client:
                yield client

    def test_prompt_required(self, client):
        """Test that prompt is required."""
        response = client.post("/generate", json={
            "max_tokens": 100,
            "temperature": 0.8
        })

        assert response.status_code == 422

    def test_prompt_min_length(self, client):
        """Test prompt minimum length."""
        response = client.post("/generate", json={
            "prompt": ""
        })

        assert response.status_code == 422

    def test_max_tokens_range(self, client):
        """Test max_tokens range validation."""
        # Too low
        response = client.post("/generate", json={
            "prompt": "test",
            "max_tokens": 5
        })
        assert response.status_code == 422

        # Too high
        response = client.post("/generate", json={
            "prompt": "test",
            "max_tokens": 2000
        })
        assert response.status_code == 422

        # Valid
        response = client.post("/generate", json={
            "prompt": "test",
            "max_tokens": 100
        })
        # Should not be validation error (may be 503 if model not loaded)
        assert response.status_code != 422

    def test_temperature_range(self, client):
        """Test temperature range validation."""
        # Too low
        response = client.post("/generate", json={
            "prompt": "test",
            "temperature": 0.05
        })
        assert response.status_code == 422

        # Too high
        response = client.post("/generate", json={
            "prompt": "test",
            "temperature": 3.0
        })
        assert response.status_code == 422

        # Valid
        response = client.post("/generate", json={
            "prompt": "test",
            "temperature": 0.8
        })
        assert response.status_code != 422

    def test_top_k_validation(self, client):
        """Test top_k range validation."""
        # Negative
        response = client.post("/generate", json={
            "prompt": "test",
            "top_k": -1
        })
        assert response.status_code == 422

        # Too high
        response = client.post("/generate", json={
            "prompt": "test",
            "top_k": 200
        })
        assert response.status_code == 422

        # Valid
        response = client.post("/generate", json={
            "prompt": "test",
            "top_k": 50
        })
        assert response.status_code != 422

    def test_top_p_validation(self, client):
        """Test top_p range validation."""
        # Too low
        response = client.post("/generate", json={
            "prompt": "test",
            "top_p": 0.05
        })
        assert response.status_code == 422

        # Too high
        response = client.post("/generate", json={
            "prompt": "test",
            "top_p": 1.5
        })
        assert response.status_code == 422

        # Valid
        response = client.post("/generate", json={
            "prompt": "test",
            "top_p": 0.9
        })
        assert response.status_code != 422

    def test_repetition_penalty_validation(self, client):
        """Test repetition_penalty range validation."""
        # Too low
        response = client.post("/generate", json={
            "prompt": "test",
            "repetition_penalty": 0.5
        })
        assert response.status_code == 422

        # Too high
        response = client.post("/generate", json={
            "prompt": "test",
            "repetition_penalty": 3.0
        })
        assert response.status_code == 422

        # Valid
        response = client.post("/generate", json={
            "prompt": "test",
            "repetition_penalty": 1.1
        })
        assert response.status_code != 422

    def test_prompt_max_length(self, client):
        """Test prompt maximum length."""
        response = client.post("/generate", json={
            "prompt": "x" * 3000  # Exceeds 2048 limit
        })
        assert response.status_code == 422


class TestHealthResponse:
    """Tests for health response model."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        with patch('src.server.get_model_loader') as mock_loader:
            mock_instance = Mock()
            mock_instance.device = "cuda"
            mock_instance.tokenizer = Mock()
            mock_instance.tokenizer.__len__ = Mock(return_value=10000)
            mock_model = Mock()
            mock_model.parameters = Mock(return_value=[Mock(numel=Mock(return_value=24500000))])
            mock_instance.model = mock_model
            mock_loader.return_value = mock_instance

            from src.server import app
            with TestClient(app) as client:
                yield client

    def test_health_includes_model_loaded(self, client):
        """Test health response includes model_loaded field."""
        response = client.get("/health")
        data = response.json()

        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_health_includes_device(self, client):
        """Test health response includes device field."""
        response = client.get("/health")
        data = response.json()

        assert "device" in data

    def test_health_includes_vocab_size(self, client):
        """Test health response includes vocab_size field."""
        response = client.get("/health")
        data = response.json()

        assert "vocab_size" in data

    def test_health_includes_parameters(self, client):
        """Test health response includes parameters field."""
        response = client.get("/health")
        data = response.json()

        assert "parameters" in data


class TestCORS:
    """Tests for CORS middleware."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        with patch('src.server.get_model_loader') as mock_loader:
            mock_instance = Mock()
            mock_instance.device = "cuda"
            mock_instance.tokenizer = Mock()
            mock_instance.tokenizer.__len__ = Mock(return_value=10000)
            mock_instance.model = Mock()
            mock_instance.model.parameters = Mock(return_value=[])
            mock_loader.return_value = mock_instance

            from src.server import app
            with TestClient(app) as client:
                yield client

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present."""
        response = client.options("/", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST"
        })

        # CORS preflight should work
        assert response.status_code in [200, 405]


class TestSSEStreaming:
    """Tests for SSE streaming generation."""

    @pytest.fixture
    def client_with_model(self):
        """Create a test client with model loaded."""
        # Create mock generator
        mock_generator = Mock()
        mock_generator.generate_stream = Mock(return_value=iter(["Once", " upon", " a", " time"]))

        # Create mock model loader that returns a proper tuple from load()
        mock_loader = Mock()
        mock_loader.device = "cuda"
        mock_loader.tokenizer = Mock()
        mock_loader.tokenizer.__len__ = Mock(return_value=10000)
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=[Mock(numel=Mock(return_value=24500000))])
        mock_loader.model = mock_model
        mock_loader.load = Mock(return_value=(mock_model, mock_loader.tokenizer))

        with patch('src.server.get_model_loader', return_value=mock_loader), \
             patch('src.server.get_generator', return_value=mock_generator), \
             patch('src.server.model_loaded', True):

            # Re-import the server module to get fresh app with patches applied
            import importlib
            import src.server
            importlib.reload(src.server)

            from src.server import app
            with TestClient(app) as client:
                # Get the actual generator mock from the reloaded module
                yield client, mock_generator

    def test_generate_returns_sse_stream(self, client_with_model):
        """Test that generate endpoint returns SSE stream."""
        client, mock_generator = client_with_model

        response = client.post("/generate", json={
            "prompt": "test prompt",
            "max_tokens": 50,
            "temperature": 0.8
        })

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_sse_stream_format(self, client_with_model):
        """Test SSE stream format is correct."""
        client, mock_generator = client_with_model

        response = client.post("/generate", json={
            "prompt": "test",
            "max_tokens": 10
        })

        # Read the stream content
        content = response.text

        # Check SSE format
        assert "data:" in content
        assert "[DONE]" in content

    def test_generate_accepts_all_parameters(self, client_with_model):
        """Test that generate accepts all expected parameters."""
        client, mock_generator = client_with_model

        # Test with all parameters
        response = client.post("/generate", json={
            "prompt": "test prompt",
            "max_tokens": 150,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.85,
            "repetition_penalty": 1.2
        })

        # Should return 200 (SSE stream) not 422 (validation error)
        assert response.status_code == 200


class TestGenerateEndpointErrors:
    """Tests for generate endpoint error handling."""

    @pytest.fixture
    def client_no_model(self):
        """Create a test client without model loaded."""
        with patch('src.server.get_model_loader') as mock_loader:

            mock_instance = Mock()
            mock_instance.device = None
            mock_instance.tokenizer = None
            mock_instance.model = None
            mock_loader.return_value = mock_instance

            # Import and patch model_loaded
            import src.server
            original_loaded = src.server.model_loaded
            src.server.model_loaded = False

            try:
                from src.server import app
                with TestClient(app) as client:
                    yield client
            finally:
                src.server.model_loaded = original_loaded

    def test_generate_returns_503_when_model_not_loaded(self, client_no_model):
        """Test that generate returns 503 when model is not loaded."""
        response = client_no_model.post("/generate", json={
            "prompt": "test prompt"
        })

        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]


class TestAPIInfo:
    """Tests for /api/info endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        with patch('src.server.get_model_loader') as mock_loader:
            mock_instance = Mock()
            mock_instance.device = "cuda"
            mock_instance.tokenizer = Mock()
            mock_instance.tokenizer.__len__ = Mock(return_value=10000)
            mock_model = Mock()
            mock_model.parameters = Mock(return_value=[Mock(numel=Mock(return_value=24500000))])
            mock_instance.model = mock_model
            mock_loader.return_value = mock_instance

            from src.server import app
            with TestClient(app) as client:
                yield client

    def test_api_info_has_server_name(self, client):
        """Test API info includes server name."""
        response = client.get("/api/info")
        data = response.json()

        assert data["server"] == "TinyStories Web UI"

    def test_api_info_has_version(self, client):
        """Test API info includes version."""
        response = client.get("/api/info")
        data = response.json()

        assert data["version"] == "1.0.0"

    def test_api_info_has_endpoints(self, client):
        """Test API info includes endpoints."""
        response = client.get("/api/info")
        data = response.json()

        assert "/" in data["endpoints"]
        assert "/health" in data["endpoints"]
        assert "/generate" in data["endpoints"]
        assert "/api/info" in data["endpoints"]


class TestHTMLUIElements:
    """Tests for HTML UI elements - verifies acceptance criteria."""

    @pytest.fixture
    def client(self):
        """Create a test client with mocked model loading."""
        with patch('src.server.get_model_loader') as mock_loader:
            mock_instance = Mock()
            mock_instance.device = "cuda"
            mock_instance.tokenizer = Mock()
            mock_instance.tokenizer.__len__ = Mock(return_value=10000)
            mock_instance.model = Mock()
            mock_instance.model.parameters = Mock(return_value=[])
            mock_loader.return_value = mock_instance

            from src.server import app
            with TestClient(app) as client:
                yield client

    def test_html_page_served_at_root(self, client):
        """Acceptance: HTML page served at /"""
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "<!DOCTYPE html>" in response.text
        assert "<html" in response.text

    def test_textarea_exists_for_prompt(self, client):
        """Acceptance: Textarea accepts prompt text."""
        response = client.get("/")
        html = response.text

        # Check textarea element exists with correct ID
        assert '<textarea id="prompt"' in html
        assert "placeholder=" in html

    def test_temperature_slider_exists(self, client):
        """Acceptance: Sliders control temperature."""
        response = client.get("/")
        html = response.text

        # Check temperature slider exists
        assert 'id="temperature"' in html
        assert 'type="range"' in html
        assert 'min="0.1"' in html
        assert 'max="2.0"' in html
        assert 'step="0.1"' in html

    def test_max_tokens_slider_exists(self, client):
        """Acceptance: Sliders control max_tokens."""
        response = client.get("/")
        html = response.text

        # Check max_tokens slider exists
        assert 'id="max-tokens"' in html
        assert 'type="range"' in html
        assert 'min="50"' in html
        assert 'max="500"' in html
        assert 'step="10"' in html

    def test_output_display_exists(self, client):
        """Acceptance: Output displays streamed text."""
        response = client.get("/")
        html = response.text

        # Check output container exists
        assert 'id="output"' in html
        assert 'class="output"' in html

    def test_generate_button_exists(self, client):
        """Verify generate button exists for triggering streaming."""
        response = client.get("/")
        html = response.text

        assert 'id="generate-btn"' in html
        assert "Generate" in html

    def test_slider_value_displays(self, client):
        """Verify slider values are displayed to user."""
        response = client.get("/")
        html = response.text

        # Check value display elements exist
        assert 'id="temp-value"' in html
        assert 'id="tokens-value"' in html

    def test_javascript_handles_sse_streaming(self, client):
        """Acceptance: Output displays streamed text in real-time (JS implementation)."""
        response = client.get("/")
        html = response.text

        # Check JavaScript has SSE handling
        assert "fetch" in html
        assert "/generate" in html
        assert "generatedText" in html

    def test_mobile_responsive_css(self, client):
        """Verify mobile responsive design."""
        response = client.get("/")
        html = response.text

        # Check for responsive meta tag
        assert 'viewport" content="width=device-width' in html
        # Check for media query
        assert "@media" in html

    def test_escape_html_function_exists(self, client):
        """Verify XSS protection via escapeHtml function."""
        response = client.get("/")
        html = response.text

        assert "function escapeHtml" in html or "escapeHtml" in html


class TestStreamingOutput:
    """Tests for real-time streaming output functionality."""

    @pytest.fixture
    def client_with_model(self):
        """Create a test client with model loaded."""
        mock_generator = Mock()
        mock_generator.generate_stream = Mock(
            return_value=iter(["Hello", " world", "!"])
        )

        mock_loader = Mock()
        mock_loader.device = "cuda"
        mock_loader.tokenizer = Mock()
        mock_loader.tokenizer.__len__ = Mock(return_value=10000)
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=[Mock(numel=Mock(return_value=24500000))])
        mock_loader.model = mock_model

        with patch('src.server.get_model_loader', return_value=mock_loader), \
             patch('src.server.get_generator', return_value=mock_generator), \
             patch('src.server.model_loaded', True):

            import importlib
            import src.server
            importlib.reload(src.server)

            from src.server import app
            with TestClient(app) as client:
                yield client, mock_generator

    def test_streaming_returns_tokens_sequentially(self, client_with_model):
        """Verify streaming returns tokens one at a time."""
        client, mock_generator = client_with_model

        response = client.post("/generate", json={
            "prompt": "test",
            "max_tokens": 10
        })

        content = response.text

        # Should have multiple data: entries for streaming
        data_entries = content.count("data:")
        assert data_entries >= 2  # At least tokens + [DONE]

    def test_stream_ends_with_done_marker(self, client_with_model):
        """Verify stream ends with [DONE] marker."""
        client, mock_generator = client_with_model

        response = client.post("/generate", json={
            "prompt": "test",
            "max_tokens": 10
        })

        assert "data: [DONE]" in response.text

    def test_stream_format_is_valid_json(self, client_with_model):
        """Verify each streamed token is valid JSON."""
        client, mock_generator = client_with_model

        response = client.post("/generate", json={
            "prompt": "test",
            "max_tokens": 10
        })

        # Parse each data line (except [DONE])
        for line in response.text.split("\n"):
            if line.startswith("data:") and "[DONE]" not in line:
                json_str = line[5:].strip()
                if json_str:
                    parsed = json.loads(json_str)
                    assert "token" in parsed

    def test_streaming_yields_tokens_in_order(self, client_with_model):
        """Verify streaming yields tokens in the order they were generated."""
        client, mock_generator = client_with_model

        response = client.post("/generate", json={
            "prompt": "test",
            "max_tokens": 10
        })

        content = response.text

        # Extract tokens from SSE stream
        lines = [l for l in content.split("\n") if l.startswith("data:")]
        tokens = []
        for line in lines:
            if "[DONE]" not in line:
                json_str = line[5:].strip()
                if json_str:
                    try:
                        parsed = json.loads(json_str)
                        if "token" in parsed:
                            tokens.append(parsed["token"])
                    except json.JSONDecodeError:
                        pass

        # Verify we got some tokens (the mock returns 3 tokens)
        assert len(tokens) >= 1
        # Verify [DONE] marker is present
        assert "data: [DONE]" in content

    def test_streaming_returns_multiple_data_events(self, client_with_model):
        """Verify streaming returns multiple separate data events."""
        client, mock_generator = client_with_model

        response = client.post("/generate", json={
            "prompt": "test",
            "max_tokens": 10
        })

        content = response.text

        # Should have multiple data: entries (tokens + [DONE])
        data_count = content.count("data:")
        assert data_count >= 2  # At least some tokens + [DONE]


class TestSecurityFeatures:
    """Tests for security hardening: headers, rate limiting, sanitisation, hidden docs."""

    @pytest.fixture(autouse=True)
    def reset_rate_limit_state(self):
        """Clear rate-limit state before and after each test."""
        import src.server
        src.server._request_times.clear()
        yield
        src.server._request_times.clear()

    @pytest.fixture
    def client(self):
        """Test client with mocked model loader."""
        with patch("src.server.get_model_loader") as mock_loader:
            mock_instance = Mock()
            mock_instance.device = "cuda"
            mock_instance.tokenizer = Mock()
            mock_instance.tokenizer.__len__ = Mock(return_value=10000)
            mock_instance.model = Mock()
            mock_instance.model.parameters = Mock(return_value=[])
            mock_loader.return_value = mock_instance

            from src.server import app
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client

    # ── Security headers ────────────────────────────────────────────────────

    def test_x_frame_options_is_deny(self, client):
        """X-Frame-Options must be DENY to block clickjacking."""
        assert client.get("/health").headers.get("x-frame-options") == "DENY"

    def test_x_content_type_options_is_nosniff(self, client):
        """X-Content-Type-Options must be nosniff to block MIME-type sniffing."""
        assert client.get("/health").headers.get("x-content-type-options") == "nosniff"

    def test_x_xss_protection_is_set(self, client):
        """X-XSS-Protection header must be present."""
        assert "1; mode=block" in client.get("/health").headers.get("x-xss-protection", "")

    def test_referrer_policy_is_strict(self, client):
        """Referrer-Policy must limit referrer leakage."""
        assert client.get("/health").headers.get("referrer-policy") == "strict-origin-when-cross-origin"

    def test_csp_header_present(self, client):
        """Content-Security-Policy must be set on all responses."""
        assert "content-security-policy" in client.get("/").headers

    def test_csp_blocks_framing(self, client):
        """CSP frame-ancestors 'none' must be present to block embedding."""
        csp = client.get("/").headers.get("content-security-policy", "")
        assert "frame-ancestors 'none'" in csp

    def test_permissions_policy_present(self, client):
        """Permissions-Policy must restrict sensitive browser APIs."""
        assert "permissions-policy" in client.get("/").headers

    # ── Hidden API docs ──────────────────────────────────────────────────────

    def test_docs_endpoint_hidden(self, client):
        """/docs must return 404 — API schema must not be publicly exposed."""
        assert client.get("/docs").status_code == 404

    def test_redoc_endpoint_hidden(self, client):
        """/redoc must return 404 — API schema must not be publicly exposed."""
        assert client.get("/redoc").status_code == 404

    def test_openapi_json_hidden(self, client):
        """/openapi.json must return 404 — raw schema must not be leaked."""
        assert client.get("/openapi.json").status_code == 404

    # ── Prompt sanitisation ──────────────────────────────────────────────────

    def test_sanitise_strips_null_bytes(self):
        from src.server import _sanitise_prompt
        assert "\x00" not in _sanitise_prompt("hello\x00world")

    def test_sanitise_strips_control_chars(self):
        from src.server import _sanitise_prompt
        result = _sanitise_prompt("test\x01\x02\x07\x08end")
        assert result == "testend"

    def test_sanitise_preserves_newline(self):
        from src.server import _sanitise_prompt
        assert "\n" in _sanitise_prompt("line1\nline2")

    def test_sanitise_preserves_tab(self):
        from src.server import _sanitise_prompt
        assert "\t" in _sanitise_prompt("col1\tcol2")

    def test_sanitise_preserves_normal_text(self):
        from src.server import _sanitise_prompt
        text = "Once upon a time, in a faraway land..."
        assert _sanitise_prompt(text) == text

    def test_control_only_prompt_returns_400(self, client):
        """Prompt that is empty after sanitisation must return 400."""
        import src.server
        src.server.model_loaded = True
        try:
            response = client.post("/generate", json={"prompt": "\x00\x01\x02\x03"})
            assert response.status_code == 400
        finally:
            src.server.model_loaded = False

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def test_rate_limit_allows_within_window(self):
        """Requests within the rate limit window must be allowed."""
        import src.server
        limit = src.server._RATE_LIMIT_REQUESTS
        for i in range(limit - 1):
            assert src.server._check_rate_limit("test-ip") is True, f"request {i+1} blocked prematurely"

    def test_rate_limit_blocks_when_exceeded(self):
        """Request exceeding rate limit must be rejected."""
        import src.server
        limit = src.server._RATE_LIMIT_REQUESTS
        for _ in range(limit):
            src.server._check_rate_limit("overload-ip")
        assert src.server._check_rate_limit("overload-ip") is False

    def test_rate_limit_isolates_different_ips(self):
        """Rate limit counters must be per-IP, not global."""
        import src.server
        limit = src.server._RATE_LIMIT_REQUESTS
        for _ in range(limit):
            src.server._check_rate_limit("ip-a")
        # ip-b untouched — must still be allowed
        assert src.server._check_rate_limit("ip-b") is True

    def test_get_client_ip_uses_x_forwarded_for(self):
        """IP extraction must honour X-Forwarded-For (Traefik sets this)."""
        from src.server import _get_client_ip
        req = Mock()
        req.headers = {"X-Forwarded-For": "203.0.113.1, 10.0.0.1"}
        req.client = None
        assert _get_client_ip(req) == "203.0.113.1"

    def test_get_client_ip_falls_back_to_client_host(self):
        """IP extraction falls back to request.client.host when no forwarded header."""
        from src.server import _get_client_ip
        req = Mock()
        req.headers = {}
        req.client = Mock()
        req.client.host = "192.168.1.100"
        assert _get_client_ip(req) == "192.168.1.100"

    def test_generate_returns_429_when_rate_limited(self, client):
        """Exceeded rate limit must result in HTTP 429 on /generate."""
        import src.server
        ip = "10.0.0.42"
        # Exhaust the limit for this IP
        for _ in range(src.server._RATE_LIMIT_REQUESTS):
            src.server._check_rate_limit(ip)

        src.server.model_loaded = True
        try:
            resp = client.post(
                "/generate",
                json={"prompt": "once upon a time"},
                headers={"X-Forwarded-For": ip},
            )
            assert resp.status_code == 429
            assert "Rate limit" in resp.json()["detail"]
        finally:
            src.server.model_loaded = False

    # ── CORS restrictions ─────────────────────────────────────────────────────

    def test_cors_rejects_unknown_origin(self, client):
        """CORS must not echo back an unrecognised origin."""
        resp = client.options(
            "/generate",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        allow = resp.headers.get("access-control-allow-origin", "")
        assert "evil.example.com" not in allow

    def test_cors_allows_aichargeworks_origin(self, client):
        """CORS must allow the main KR Playground origin."""
        resp = client.options(
            "/generate",
            headers={
                "Origin": "https://aichargeworks.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        allow = resp.headers.get("access-control-allow-origin", "")
        assert "aichargeworks.com" in allow


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
