"""FastAPI server for TinyStories model with SSE streaming.

This module provides:
1. FastAPI server on port 7779
2. SSE streaming /generate endpoint
3. Model loading on startup via lifespan
4. Health check endpoint
"""

import os
import sys
import time
import logging
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Any
from contextlib import asynccontextmanager
from collections import defaultdict
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import get_model_loader, get_generator, ModelLoader, StreamingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state with explicit type annotation
model_loaded: bool = False

# Rate limiting state (in-memory, sliding window, no extra deps)
_request_times: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_REQUESTS: int = 10    # max requests per IP per window
_RATE_LIMIT_WINDOW: int = 60      # seconds
_MAX_CONCURRENT: int = 3          # max simultaneous generation streams
_generation_semaphore: asyncio.Semaphore = asyncio.Semaphore(_MAX_CONCURRENT)


def _get_client_ip(http_request: Request) -> str:
    """Extract real client IP, honouring Traefik X-Forwarded-For header."""
    forwarded = http_request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return http_request.client.host if http_request.client else "unknown"


def _check_rate_limit(ip: str) -> bool:
    """Return True if request is within rate limit, False if exceeded.

    Uses a sliding-window approach: only counts requests within the last
    _RATE_LIMIT_WINDOW seconds.
    """
    now = time.time()
    window_start = now - _RATE_LIMIT_WINDOW
    # Purge old timestamps
    _request_times[ip] = [t for t in _request_times[ip] if t > window_start]
    if len(_request_times[ip]) >= _RATE_LIMIT_REQUESTS:
        return False
    _request_times[ip].append(now)
    return True


# Request/Response models
class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., min_length=1, max_length=2048, description="Input prompt text")
    max_tokens: int = Field(default=100, ge=10, le=1000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=50, ge=0, le=100, description="Top-k sampling")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Nucleus sampling threshold")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str | None = None
    vocab_size: int | None = None
    parameters: str | None = None


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model_loaded

    logger.info("=" * 60)
    logger.info("TinyStories Web UI Server Starting")
    logger.info("=" * 60)

    # Load model on startup
    try:
        loader = get_model_loader()

        # Get paths from environment or use defaults
        checkpoint_path = os.environ.get(
            "CHECKPOINT_PATH",
            "checkpoints/checkpoint_best_loss.pth"
        )
        tokenizer_path = os.environ.get(
            "TOKENIZER_PATH",
            "tokenizer/tinystories_10k"
        )
        device = os.environ.get("DEVICE", "cuda")

        logger.info(f"Loading model from: {checkpoint_path}")
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        logger.info(f"Target device: {device}")

        model, tokenizer = loader.load(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            device=device,
        )

        model_loaded = True
        logger.info("✅ Model loaded successfully!")

        # Log model info
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Parameters: {params/1e6:.2f}M")
        logger.info(f"   Vocab size: {len(tokenizer)}")
        logger.info(f"   Device: {loader.device}")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        model_loaded = False
        # Continue anyway - health endpoint will report status

    logger.info("=" * 60)
    logger.info("Server ready!")
    logger.info("=" * 60)

    yield

    # Cleanup on shutdown
    logger.info("Server shutting down...")


# Generation timeout — prevents a single request holding a semaphore slot forever
_GENERATION_TIMEOUT_SECONDS: int = 60

# Allowed CORS origins — restrict to our own domains
_ALLOWED_ORIGINS: list[str] = [
    "https://aichargeworks.com",
    "https://www.aichargeworks.com",
    "https://tinystories.aichargeworks.com",
]

import re as _re

# Prompt sanitisation — reject control chars except tab/newline
_CONTROL_CHAR_PATTERN = _re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitise_prompt(prompt: str) -> str:
    """Strip dangerous control characters from prompt text."""
    return _CONTROL_CHAR_PATTERN.sub("", prompt).strip()


# Create FastAPI app with lifespan
app = FastAPI(
    title="TinyStories Web UI",
    description="Web interface for TinyStories story generation model with streaming SSE",
    version="1.0.0",
    lifespan=lifespan,
    # Hide API docs on production to reduce attack surface
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Add CORS middleware — restrict to known origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security response headers on every response."""

    async def dispatch(self, request: Request, call_next: Any) -> StarletteResponse:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        # CSP: allow fonts from Google + self only; no inline scripts (the template uses defer/inline — allow unsafe-inline for UI only)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; "
            "img-src 'self' https://aichargeworks.com data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)


# HTML template for the UI — matches KR Playground glassmorphic theme
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TinyStories — Live Inference</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="icon" type="image/webp" href="https://aichargeworks.com/kr-logo.webp">
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
            --glass-hover: rgba(255, 255, 255, 0.06);
            --text-primary: #ffffff;
            --text-secondary: rgba(255, 255, 255, 0.6);
            --text-tertiary: rgba(255, 255, 255, 0.4);
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-green: #10b981;
            --accent-orange: #f59e0b;
            --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            line-height: 1.6;
        }

        .bg-gradient {
            position: fixed; top: 0; left: 0;
            width: 100%; height: 100%;
            z-index: 0; overflow: hidden;
        }

        .gradient-orb {
            position: absolute; border-radius: 50%;
            filter: blur(100px); opacity: 0.4;
            animation: float 25s ease-in-out infinite;
        }

        .orb-1 { width: 800px; height: 800px; background: var(--gradient-1); top: -400px; left: -200px; }
        .orb-2 { width: 600px; height: 600px; background: var(--gradient-2); bottom: -200px; right: -100px; animation-delay: -8s; }
        .orb-3 { width: 500px; height: 500px; background: var(--gradient-3); top: 50%; left: 50%; transform: translate(-50%,-50%); animation-delay: -16s; opacity: 0.2; }

        @keyframes float {
            0%, 100% { transform: translate(0,0) scale(1); }
            25% { transform: translate(60px,-80px) scale(1.1); }
            50% { transform: translate(-40px,60px) scale(0.95); }
            75% { transform: translate(80px,40px) scale(1.05); }
        }

        /* orb-3 has its own base transform */
        .orb-3 { animation-name: float3; }
        @keyframes float3 {
            0%, 100% { transform: translate(-50%,-50%) scale(1); }
            25% { transform: translate(calc(-50% + 60px), calc(-50% - 80px)) scale(1.1); }
            50% { transform: translate(calc(-50% - 40px), calc(-50% + 60px)) scale(0.95); }
            75% { transform: translate(calc(-50% + 80px), calc(-50% + 40px)) scale(1.05); }
        }

        .grid-pattern {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background-image:
                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 60px 60px;
            z-index: 1; pointer-events: none;
        }

        .container {
            position: relative; z-index: 2;
            max-width: 860px;
            margin: 0 auto;
            padding: 0 24px 80px;
        }

        nav {
            padding: 24px 0;
            display: flex; justify-content: space-between; align-items: center;
        }

        .logo {
            display: flex; align-items: center; gap: 10px;
            font-size: 1.25rem; font-weight: 700;
            background: linear-gradient(135deg, #fff 0%, #a8edea 50%, #fed6e3 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; text-decoration: none;
        }

        .logo img { height: 32px; width: auto; filter: drop-shadow(0 0 8px rgba(100,255,218,0.4)); }

        .nav-back {
            color: var(--text-secondary); text-decoration: none;
            font-size: 0.85rem; font-weight: 500;
            display: flex; align-items: center; gap: 6px;
            transition: color 0.2s ease;
        }
        .nav-back:hover { color: var(--text-primary); }

        .hero {
            padding: 56px 0 40px;
            text-align: center;
        }

        .hero-badge {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 7px 16px;
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 50px;
            font-size: 0.8rem; color: var(--text-secondary);
            margin-bottom: 24px;
        }

        .hero-badge .dot {
            width: 7px; height: 7px; border-radius: 50%;
            background: #555; transition: background 0.3s;
        }

        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

        .hero-title {
            font-size: clamp(2.2rem, 6vw, 3.8rem);
            font-weight: 800; letter-spacing: -1.5px; line-height: 1.1;
            margin-bottom: 14px;
        }

        .gradient-text {
            background: linear-gradient(135deg, #fff 0%, #a8edea 40%, #fed6e3 70%, #ffecd2 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero-subtitle {
            font-size: 1rem; color: var(--text-secondary); margin-bottom: 28px;
        }

        .stats-row {
            display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;
            margin-bottom: 48px;
        }

        .stat-chip {
            display: inline-flex; align-items: center; gap: 5px;
            padding: 5px 13px;
            background: var(--glass-bg); border: 1px solid var(--glass-border);
            border-radius: 8px;
            font-size: 0.76rem; font-weight: 500; color: var(--text-secondary);
        }

        .card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 20px; padding: 32px;
            margin-bottom: 20px;
            position: relative; overflow: hidden;
        }

        .card::before {
            content: ''; position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: var(--gradient-1);
        }

        textarea {
            width: 100%;
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            font-size: 0.95rem; line-height: 1.6;
            padding: 16px; resize: vertical; outline: none;
            transition: border-color 0.2s ease;
            margin-bottom: 20px; min-height: 90px;
        }

        textarea:focus { border-color: rgba(99,102,241,0.5); }
        textarea::placeholder { color: var(--text-tertiary); }

        .controls {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 20px; margin-bottom: 24px;
        }

        .slider-group { display: flex; flex-direction: column; gap: 10px; }

        .slider-label {
            display: flex; justify-content: space-between;
            font-size: 0.82rem; color: var(--text-secondary);
        }

        .slider-label .val {
            color: var(--accent-blue); font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        input[type="range"] { width: 100%; accent-color: var(--accent-blue); cursor: pointer; }

        .actions {
            display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
        }

        .btn {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 12px 24px; border-radius: 12px;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem; font-weight: 600;
            cursor: pointer; border: none; transition: all 0.3s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: white;
        }

        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(99,102,241,0.35); }

        .status-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--accent-green); flex-shrink: 0;
            transition: background 0.2s;
        }

        .status-dot.generating { background: var(--accent-orange); animation: pulse 0.8s ease-in-out infinite; }
        .status-dot.error { background: #ef4444; }

        .status-label { font-size: 0.82rem; color: var(--text-tertiary); }

        .error-banner {
            display: none; align-items: center; gap: 10px;
            padding: 12px 16px;
            background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2);
            border-radius: 10px;
            font-size: 0.82rem; color: rgba(252,165,165,0.9);
            margin-bottom: 16px;
        }

        .output-card { min-height: 180px; }

        .output-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 16px;
        }

        .output-label { font-size: 0.82rem; font-weight: 600; color: var(--text-secondary); }

        .copy-btn {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 12px; border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-size: 0.76rem; font-weight: 500;
            background: var(--glass-bg); border: 1px solid var(--glass-border);
            color: var(--text-secondary); cursor: pointer; transition: all 0.2s ease;
        }

        .copy-btn:hover { background: var(--glass-hover); color: var(--text-primary); }

        .output-area {
            background: rgba(0,0,0,0.4); border: 1px solid var(--glass-border);
            border-radius: 12px; padding: 20px;
            min-height: 140px; max-height: 400px; overflow-y: auto;
        }

        .prompt-echo {
            color: var(--text-tertiary);
            font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
            margin-bottom: 10px; padding-bottom: 10px;
            border-bottom: 1px solid var(--glass-border);
            display: none;
        }

        #generated-text {
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace; font-size: 0.88rem;
            line-height: 1.8; white-space: pre-wrap; word-break: break-word;
        }

        .placeholder-text { color: var(--text-tertiary); font-size: 0.88rem; }

        footer { text-align: center; padding: 32px 0 0; }
        .footer-text { font-size: 0.78rem; color: var(--text-tertiary); }
        .footer-text a { color: var(--text-secondary); text-decoration: none; }
        .footer-text a:hover { color: var(--text-primary); }

        @media (max-width: 640px) {
            .controls { grid-template-columns: 1fr; }
            .card { padding: 20px; }
            .hero { padding: 36px 0 28px; }
        }
    </style>
</head>
<body>
    <div class="bg-gradient">
        <div class="gradient-orb orb-1"></div>
        <div class="gradient-orb orb-2"></div>
        <div class="gradient-orb orb-3"></div>
    </div>
    <div class="grid-pattern"></div>

    <div class="container">
        <nav>
            <a href="https://aichargeworks.com" class="logo">
                <img src="https://aichargeworks.com/kr-logo.webp" alt="KR" onerror="this.style.display='none'">
                KR Playground
            </a>
            <a href="https://aichargeworks.com" class="nav-back">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M19 12H5M12 5l-7 7 7 7"/>
                </svg>
                Back to Playground
            </a>
        </nav>

        <div class="hero">
            <div class="hero-badge">
                <span class="dot" id="model-dot"></span>
                <span id="model-status-text">Checking model&hellip;</span>
            </div>
            <h1 class="hero-title"><span class="gradient-text">TinyStories</span> Generator</h1>
            <p class="hero-subtitle">24.5M parameter Transformer trained from scratch &mdash; live inference</p>
            <div class="stats-row">
                <span class="stat-chip">&#x1F4C9; PPL 8.65</span>
                <span class="stat-chip">&#x1F4E6; 24.5M params</span>
                <span class="stat-chip">&#x1F524; 10K vocab</span>
                <span class="stat-chip">&#x26A1; RTX 4070 Super</span>
                <span class="stat-chip">&#x1F3D7; Custom Transformer</span>
            </div>
        </div>

        <div class="card">
            <div class="error-banner" id="error-banner">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
                </svg>
                Model offline &mdash; try again in a moment.
            </div>

            <textarea id="prompt" placeholder="Once upon a time, there was a little girl named Lily..." rows="3"></textarea>

            <div class="controls">
                <div class="slider-group">
                    <div class="slider-label">
                        Temperature
                        <span class="val" id="temp-val">0.80</span>
                    </div>
                    <input type="range" id="temperature" min="0.1" max="2.0" step="0.05" value="0.8">
                </div>
                <div class="slider-group">
                    <div class="slider-label">
                        Max Tokens
                        <span class="val" id="tokens-val">200</span>
                    </div>
                    <input type="range" id="max-tokens" min="50" max="500" step="10" value="200">
                </div>
            </div>

            <div class="actions">
                <button id="generate-btn" class="btn btn-primary" onclick="generate()">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
                    Generate Story
                </button>
                <span class="status-dot" id="status-dot"></span>
                <span class="status-label" id="status-label">Ready</span>
            </div>
        </div>

        <div class="card output-card">
            <div class="output-header">
                <span class="output-label">Generated Story</span>
                <button class="copy-btn" id="copy-btn" onclick="copyOutput()">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2"/>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                    </svg>
                    Copy
                </button>
            </div>
            <div class="output-area" id="output-area">
                <div class="prompt-echo" id="prompt-echo"></div>
                <div id="generated-text"></div>
                <div class="placeholder-text" id="placeholder">Your generated story will appear here&hellip;</div>
            </div>
        </div>

        <footer>
            <p class="footer-text">
                Built by <a href="https://aichargeworks.com">KR</a> &bull;
                <a href="https://huggingface.co/karthyick/tinystories-24.5m-article-generation" target="_blank">View on HuggingFace</a> &bull;
                <a href="https://aichargeworks.com">KR Playground</a>
            </p>
        </footer>
    </div>

    <script>
        // Slider live updates
        document.getElementById('temperature').addEventListener('input', e => {
            document.getElementById('temp-val').textContent = parseFloat(e.target.value).toFixed(2);
        });
        document.getElementById('max-tokens').addEventListener('input', e => {
            document.getElementById('tokens-val').textContent = e.target.value;
        });

        let controller = null;
        let running = false;

        function setStatus(state, msg) {
            const dot = document.getElementById('status-dot');
            dot.className = 'status-dot' + (state === 'generating' ? ' generating' : state === 'error' ? ' error' : '');
            document.getElementById('status-label').textContent = msg;
        }

        async function generate() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) return;

            if (running) { controller.abort(); return; }

            const temperature = parseFloat(document.getElementById('temperature').value);
            const max_tokens = parseInt(document.getElementById('max-tokens').value);

            running = true;
            controller = new AbortController();

            const btn = document.getElementById('generate-btn');
            btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12"/></svg> Stop';

            const promptEcho = document.getElementById('prompt-echo');
            promptEcho.textContent = '> ' + prompt;
            promptEcho.style.display = 'block';
            document.getElementById('placeholder').style.display = 'none';
            document.getElementById('generated-text').textContent = '';
            document.getElementById('error-banner').style.display = 'none';
            setStatus('generating', 'Generating\u2026');

            try {
                const resp = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, max_tokens, temperature, top_k: 50, top_p: 0.9, repetition_penalty: 1.1 }),
                    signal: controller.signal,
                });

                if (!resp.ok) throw new Error('HTTP ' + resp.status);

                const reader = resp.body.getReader();
                const dec = new TextDecoder();
                const textEl = document.getElementById('generated-text');
                const area = document.getElementById('output-area');
                let buf = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buf += dec.decode(value, { stream: true });
                    const lines = buf.split('\\n');
                    buf = lines.pop();
                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const data = line.slice(6);
                        if (data === '[DONE]') break;
                        try {
                            const json = JSON.parse(data);
                            if (json.token) textEl.textContent += json.token;
                        } catch {}
                    }
                    area.scrollTop = area.scrollHeight;
                }
                setStatus('ready', 'Complete \u2713');
            } catch (err) {
                if (err.name === 'AbortError') {
                    setStatus('ready', 'Stopped');
                } else {
                    setStatus('error', 'Error');
                    const banner = document.getElementById('error-banner');
                    banner.style.display = 'flex';
                    setTimeout(() => banner.style.display = 'none', 8000);
                }
            } finally {
                running = false;
                controller = null;
                btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> Generate Story';
            }
        }

        function copyOutput() {
            const text = document.getElementById('generated-text').textContent;
            if (!text) return;
            navigator.clipboard.writeText(text).then(() => {
                const btn = document.getElementById('copy-btn');
                btn.textContent = 'Copied!';
                setTimeout(() => {
                    btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';
                }, 2000);
            });
        }

        // Enter to generate (Shift+Enter = newline)
        document.getElementById('prompt').addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); generate(); }
        });

        // Health check on load
        (async () => {
            const dot = document.getElementById('model-dot');
            const txt = document.getElementById('model-status-text');
            try {
                const r = await fetch('/health', { signal: AbortSignal.timeout(5000) });
                const d = await r.json();
                if (d.model_loaded) {
                    dot.style.cssText = 'background:#10b981;animation:pulse 2s ease-in-out infinite';
                    txt.textContent = 'Model loaded';
                    setStatus('ready', 'Ready');
                } else {
                    dot.style.background = '#f59e0b';
                    txt.textContent = 'Model not loaded';
                    setStatus('error', 'Model not loaded');
                }
            } catch {
                dot.style.background = '#ef4444';
                txt.textContent = 'Server offline';
                setStatus('error', 'Offline');
                document.getElementById('error-banner').style.display = 'flex';
            }
        })();
    </script>
</body>
</html>
"""


async def generate_sse_stream(
    generator: StreamingGenerator,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> AsyncGenerator[str, None]:
    """Generate SSE events from the model with TRUE real-time streaming.

    Uses asyncio.Queue to stream tokens as they're generated, not after all
    generation completes. This provides real-time feedback to users.

    Args:
        generator: StreamingGenerator instance
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        repetition_penalty: Repetition penalty

    Yields:
        SSE-formatted strings as tokens are generated
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _generate_to_queue() -> None:
        """Generate tokens and push to queue for real-time streaming."""
        try:
            for token in generator.generate_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            ):
                # Put token in queue for async consumption
                queue.put_nowait(token)
            # Signal completion
            queue.put_nowait(None)
        except Exception as e:
            # Put error in queue
            queue.put_nowait(f"__ERROR__:{str(e)}")

    # Start generation in thread pool
    loop.run_in_executor(None, _generate_to_queue)

    # Stream tokens from queue as they arrive
    try:
        while True:
            token = await queue.get()

            # Check for completion signal
            if token is None:
                break

            # Check for error
            if isinstance(token, str) and token.startswith("__ERROR__:"):
                error_msg = token[10:]
                logger.error(f"Error during generation: {error_msg}")
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                return

            # Format as SSE event with JSON payload
            yield f"data: {json.dumps({'token': token})}\n\n"

        # Send completion event
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error during streaming: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the HTML UI."""
    return HTML_TEMPLATE


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server and model status."""
    loader = get_model_loader()

    response = HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        device=str(loader.device) if loader.device else None,
        vocab_size=len(loader.tokenizer) if loader.tokenizer else None,
        parameters=f"{sum(p.numel() for p in loader.model.parameters())/1e6:.2f}M" if loader.model else None,
    )

    return response


@app.post("/generate")
async def generate(request: GenerateRequest, http_request: Request):
    """Generate text with SSE streaming.

    Returns a streaming response with Server-Sent Events.
    Rate limited: 10 requests / 60s per IP, max 3 concurrent streams.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs for loading errors."
        )

    # Sanitise prompt — strip control characters
    clean_prompt = _sanitise_prompt(request.prompt)
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty after sanitisation.")

    # Per-IP rate limit check
    client_ip = _get_client_ip(http_request)
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {_RATE_LIMIT_REQUESTS} requests per {_RATE_LIMIT_WINDOW}s. Please wait.",
        )

    # Concurrent generation limit — fail fast without blocking
    if not _generation_semaphore._value:  # noqa: SLF001
        raise HTTPException(
            status_code=429,
            detail=f"Server busy: max {_MAX_CONCURRENT} concurrent generations in progress. Try again shortly.",
        )

    async def _stream_with_semaphore() -> AsyncGenerator[str, None]:
        async with _generation_semaphore:
            try:
                async with asyncio.timeout(_GENERATION_TIMEOUT_SECONDS):
                    async for chunk in generate_sse_stream(
                        generator=get_generator(),
                        prompt=clean_prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        repetition_penalty=request.repetition_penalty,
                    ):
                        yield chunk
            except TimeoutError:
                logger.warning("Generation timed out for IP %s", client_ip)
                yield "data: [DONE]\n\n"

    return StreamingResponse(
        _stream_with_semaphore(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.get("/api/info")
async def get_info():
    """Get model and server information."""
    loader = get_model_loader()

    return {
        "server": "TinyStories Web UI",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "model": {
            "parameters": f"{sum(p.numel() for p in loader.model.parameters())/1e6:.2f}M" if loader.model else None,
            "vocab_size": len(loader.tokenizer) if loader.tokenizer else None,
            "device": str(loader.device) if loader.device else None,
        },
        "endpoints": {
            "/": "HTML UI",
            "/health": "Health check",
            "/generate": "Text generation (POST, SSE streaming)",
            "/api/info": "Server information",
        },
    }


def main():
    """Run the server."""
    import uvicorn

    port = int(os.environ.get("PORT", 7779))
    host = os.environ.get("HOST", "127.0.0.1")

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "src.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
