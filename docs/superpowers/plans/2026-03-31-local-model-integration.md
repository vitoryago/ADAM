# Local Model Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic local model provider so ADAM's Deep Discussion can discover and use any OpenAI-compatible local inference server (Ollama, vLLM, MLX Server, LM Studio) alongside cloud providers.

**Architecture:** A new `LocalModelProvider` class probes configurable endpoints, discovers available models via `/v1/models`, and exposes them through a `LOCAL` provider in `AsyncLLMClient`. A "Prefer Local" toggle in Deep Discussion's config screen lets users switch between cloud and local defaults. Zero changes to existing patterns, Scratchpad, or CostGuard.

**Tech Stack:** Python 3.10+, FastAPI, AsyncOpenAI, httpx (for discovery probes), React 18 + TypeScript, VS Code Extension API

**Spec:** `docs/superpowers/specs/2026-03-31-local-model-integration-design.md`

---

### Task 1: LocalModelProvider — Discovery and Registry

**Files:**
- Create: `src/adam/llm/local_provider.py`
- Test: `tests/test_local_provider.py`

- [ ] **Step 1: Write the failing tests for LocalModelProvider**

```python
# tests/test_local_provider.py
"""Tests for LocalModelProvider: discovery, registry, health checks."""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from adam.llm.local_provider import LocalModelProvider, LocalModel


class TestLocalModelParsing:
    def test_parse_model_id_extracts_parameter_count(self):
        provider = LocalModelProvider(endpoints=[])
        model = provider._parse_model_entry("qwen3.5:72b-q4_K_M", "http://localhost:11434/v1")
        assert model.model_id == "qwen3.5:72b-q4_K_M"
        assert model.parameter_count == 72
        assert model.quantization == "q4_K_M"

    def test_parse_model_id_no_quant(self):
        provider = LocalModelProvider(endpoints=[])
        model = provider._parse_model_entry("llama3:8b", "http://localhost:11434/v1")
        assert model.parameter_count == 8
        assert model.quantization == ""

    def test_parse_model_id_no_size(self):
        provider = LocalModelProvider(endpoints=[])
        model = provider._parse_model_entry("my-custom-model", "http://localhost:5000/v1")
        assert model.parameter_count == 0
        assert model.quantization == ""

    def test_display_name_cleaned(self):
        provider = LocalModelProvider(endpoints=[])
        model = provider._parse_model_entry("qwen3.5:72b-q4_K_M", "http://localhost:11434/v1")
        assert "Qwen3.5" in model.display_name or "qwen3.5" in model.display_name.lower()
        assert "72B" in model.display_name or "72b" in model.display_name.lower()


class TestLocalModelDiscovery:
    @pytest.mark.asyncio
    async def test_discover_from_openai_compatible_endpoint(self):
        """Discovery via /v1/models returns model list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "qwen3.5:72b-q4_K_M"},
                {"id": "deepseek-coder:33b-q4_K_M"},
            ]
        }

        provider = LocalModelProvider(endpoints=["http://localhost:11434"])
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            await provider.discover()

        assert len(provider.models) == 2
        assert "qwen3.5:72b-q4_K_M" in provider.models
        assert "deepseek-coder:33b-q4_K_M" in provider.models

    @pytest.mark.asyncio
    async def test_discover_unreachable_endpoint_returns_empty(self):
        """Unreachable endpoint produces no models, no error."""
        provider = LocalModelProvider(endpoints=["http://localhost:99999"])
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=Exception("Connection refused")):
            await provider.discover()

        assert len(provider.models) == 0

    @pytest.mark.asyncio
    async def test_discover_multiple_endpoints(self):
        """Multiple backends merge into one registry."""
        resp_a = MagicMock()
        resp_a.status_code = 200
        resp_a.json.return_value = {"data": [{"id": "model-a"}]}

        resp_b = MagicMock()
        resp_b.status_code = 200
        resp_b.json.return_value = {"data": [{"id": "model-b"}]}

        call_count = 0

        async def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "11434" in url:
                return resp_a
            return resp_b

        provider = LocalModelProvider(endpoints=[
            "http://localhost:11434",
            "http://localhost:8000",
        ])
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=mock_get):
            await provider.discover()

        assert "model-a" in provider.models
        assert "model-b" in provider.models


class TestLocalModelAvailability:
    def test_get_available_models_filters_unavailable(self):
        provider = LocalModelProvider(endpoints=[])
        provider.models = {
            "good": LocalModel(
                model_id="good", display_name="Good", backend="ollama",
                base_url="http://localhost:11434/v1", parameter_count=72,
                quantization="q4_K_M", available=True,
            ),
            "bad": LocalModel(
                model_id="bad", display_name="Bad", backend="ollama",
                base_url="http://localhost:11434/v1", parameter_count=7,
                quantization="q4_K_M", available=False,
            ),
        }
        available = provider.get_available_models()
        assert len(available) == 1
        assert available[0].model_id == "good"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vitoryago/ADAM && python -m pytest tests/test_local_provider.py -v`
Expected: FAIL — `adam.llm.local_provider` does not exist

- [ ] **Step 3: Implement LocalModelProvider**

Create `src/adam/llm/local_provider.py`:

```python
"""Local Model Provider — discovers and manages locally-served LLMs.

Probes OpenAI-compatible endpoints (Ollama, vLLM, MLX Server, LM Studio)
to build a registry of available models. Backend-agnostic — works with
anything that serves ``/v1/chat/completions``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Default endpoints to probe when LOCAL_MODEL_ENDPOINTS is not set
_DEFAULT_ENDPOINTS = ["http://localhost:11434"]

# Regex to extract parameter count (e.g., "72b", "33b", "8b") from model IDs
_PARAM_RE = re.compile(r"(\d+)[bB]")

# Regex to extract quantization suffix (e.g., "q4_K_M", "q3_K_S")
_QUANT_RE = re.compile(r"(q\d+[_a-zA-Z]*)", re.IGNORECASE)


@dataclass
class LocalModel:
    """A single model served by a local inference backend."""

    model_id: str
    display_name: str
    backend: str  # "ollama" | "vllm" | "mlx" | "lmstudio" | "custom"
    base_url: str  # e.g., "http://localhost:11434/v1"
    parameter_count: int  # in billions, 0 if unknown
    quantization: str  # e.g., "q4_K_M", "" if unknown
    available: bool = True


class LocalModelProvider:
    """Discovers and tracks locally-served LLMs.

    Args:
        endpoints: Base URLs to probe (without ``/v1`` suffix).
            Defaults to ``LOCAL_MODEL_ENDPOINTS`` env var or
            ``["http://localhost:11434"]``.
    """

    def __init__(self, endpoints: Optional[List[str]] = None) -> None:
        if endpoints is None:
            raw = os.getenv("LOCAL_MODEL_ENDPOINTS", "")
            endpoints = [e.strip() for e in raw.split(",") if e.strip()] if raw else _DEFAULT_ENDPOINTS
        self.endpoints = endpoints
        self.models: Dict[str, LocalModel] = {}
        self._health_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def discover(self) -> None:
        """Probe all endpoints and build the model registry."""
        discovered: Dict[str, LocalModel] = {}
        tasks = [self._probe_endpoint(ep) for ep in self.endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, dict):
                discovered.update(result)

        self.models = discovered
        if discovered:
            logger.info(
                "Discovered %d local model(s): %s",
                len(discovered),
                ", ".join(discovered.keys()),
            )
        else:
            logger.info("No local models discovered — proceeding cloud-only")

    async def _probe_endpoint(self, base_url: str) -> Dict[str, LocalModel]:
        """Probe a single endpoint and return its models."""
        models: Dict[str, LocalModel] = {}
        v1_url = base_url.rstrip("/") + "/v1"

        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{v1_url}/models")
                if resp.status_code == 200:
                    data = resp.json()
                    model_list = data.get("data", [])
                    for entry in model_list:
                        model_id = entry.get("id", "")
                        if model_id:
                            model = self._parse_model_entry(model_id, v1_url)
                            model.backend = self._detect_backend(base_url)
                            models[model_id] = model
        except Exception as exc:
            logger.debug("Could not reach %s: %s", base_url, exc)

        return models

    def _parse_model_entry(self, model_id: str, base_url: str) -> LocalModel:
        """Parse a model ID into a LocalModel with extracted metadata."""
        param_match = _PARAM_RE.search(model_id)
        parameter_count = int(param_match.group(1)) if param_match else 0

        quant_match = _QUANT_RE.search(model_id)
        quantization = quant_match.group(1) if quant_match else ""

        display_name = self._build_display_name(model_id, parameter_count, quantization)

        return LocalModel(
            model_id=model_id,
            display_name=display_name,
            backend="custom",
            base_url=base_url,
            parameter_count=parameter_count,
            quantization=quantization,
            available=True,
        )

    @staticmethod
    def _build_display_name(model_id: str, param_count: int, quant: str) -> str:
        """Build a human-friendly display name from a model ID."""
        # Take the base name (before any colon tag)
        name = model_id.split(":")[0] if ":" in model_id else model_id
        parts = [name]
        if param_count:
            parts.append(f"{param_count}B")
        if quant:
            parts.append(f"({quant.upper()})")
        return " ".join(parts)

    @staticmethod
    def _detect_backend(base_url: str) -> str:
        """Guess the backend type from the URL port."""
        if ":11434" in base_url:
            return "ollama"
        elif ":8000" in base_url:
            return "vllm"
        elif ":5000" in base_url:
            return "mlx"
        elif ":1234" in base_url:
            return "lmstudio"
        return "custom"

    # ------------------------------------------------------------------
    # Registry access
    # ------------------------------------------------------------------

    def get_available_models(self) -> List[LocalModel]:
        """Return only models that are currently healthy."""
        return [m for m in self.models.values() if m.available]

    def has_model(self, model_id: str) -> bool:
        """Check if a model ID is in the local registry and available."""
        model = self.models.get(model_id)
        return model is not None and model.available

    def get_base_url(self, model_id: str) -> Optional[str]:
        """Return the base URL for a local model, or None."""
        model = self.models.get(model_id)
        return model.base_url if model else None

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def start_health_checks(self, interval: float = 30.0) -> None:
        """Start a background task that pings backends periodically."""
        if self._health_task is not None:
            return
        self._health_task = asyncio.create_task(self._health_loop(interval))

    async def stop_health_checks(self) -> None:
        """Cancel the background health check task."""
        if self._health_task is not None:
            self._health_task.cancel()
            self._health_task = None

    async def _health_loop(self, interval: float) -> None:
        """Periodically re-discover models to track availability."""
        while True:
            await asyncio.sleep(interval)
            try:
                await self.discover()
            except Exception as exc:
                logger.warning("Health check error: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vitoryago/ADAM && python -m pytest tests/test_local_provider.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/adam/llm/local_provider.py tests/test_local_provider.py
git commit -m "feat: add LocalModelProvider with discovery and health checks"
```

---

### Task 2: Integrate LOCAL Provider into AsyncLLMClient

**Files:**
- Modify: `src/adam/llm/async_client.py:68-73` (AsyncLLMProvider enum)
- Modify: `src/adam/llm/async_client.py:109-163` (_initialize_async_clients)
- Modify: `src/adam/llm/async_client.py:532-544` (_get_provider_for_model)
- Modify: `src/adam/llm/config.py:14-18` (ModelProvider enum)
- Test: `tests/test_local_integration.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_local_integration.py
"""Tests for LOCAL provider integration in AsyncLLMClient."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from adam.llm.async_client import AsyncLLMProvider, AsyncLLMClient
from adam.llm.config import ModelProvider
from adam.llm.local_provider import LocalModelProvider, LocalModel


class TestLocalProviderEnum:
    def test_local_in_async_provider_enum(self):
        assert AsyncLLMProvider.LOCAL.value == "local"

    def test_local_in_model_provider_enum(self):
        assert ModelProvider.LOCAL.value == "local"


class TestLocalModelRouting:
    def test_cloud_prefixes_still_route_correctly(self):
        """Existing cloud models are unaffected."""
        client = AsyncLLMClient()
        assert client._get_provider_for_model("grok-4.20-multi-agent-0309") == AsyncLLMProvider.XAI
        assert client._get_provider_for_model("claude-opus-4-6") == AsyncLLMProvider.ANTHROPIC
        assert client._get_provider_for_model("gpt-5.4-2026-03-05") == AsyncLLMProvider.OPENAI
        assert client._get_provider_for_model("gemini-3.1-pro-preview") == AsyncLLMProvider.GEMINI

    def test_local_model_routes_to_local_provider(self):
        """A model in the local registry routes to LOCAL."""
        client = AsyncLLMClient()
        # Inject a mock local provider with a known model
        mock_provider = LocalModelProvider(endpoints=[])
        mock_provider.models = {
            "qwen3.5:72b-q4_K_M": LocalModel(
                model_id="qwen3.5:72b-q4_K_M", display_name="Qwen 3.5 72B",
                backend="ollama", base_url="http://localhost:11434/v1",
                parameter_count=72, quantization="q4_K_M", available=True,
            ),
        }
        client._local_provider = mock_provider
        assert client._get_provider_for_model("qwen3.5:72b-q4_K_M") == AsyncLLMProvider.LOCAL

    def test_unknown_model_without_local_falls_back_to_xai(self):
        """Unknown model with no local match falls back to XAI (existing behavior)."""
        client = AsyncLLMClient()
        client._local_provider = LocalModelProvider(endpoints=[])
        assert client._get_provider_for_model("some-random-model") == AsyncLLMProvider.XAI


class TestLocalCostTracking:
    def test_local_model_cost_is_zero(self):
        """Local models should report zero cost."""
        client = AsyncLLMClient()
        mock_provider = LocalModelProvider(endpoints=[])
        mock_provider.models = {
            "qwen3.5:72b-q4_K_M": LocalModel(
                model_id="qwen3.5:72b-q4_K_M", display_name="Qwen 3.5 72B",
                backend="ollama", base_url="http://localhost:11434/v1",
                parameter_count=72, quantization="q4_K_M", available=True,
            ),
        }
        client._local_provider = mock_provider
        cost = client._calculate_cost("qwen3.5:72b-q4_K_M", 1000, 500)
        assert cost == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vitoryago/ADAM && python -m pytest tests/test_local_integration.py -v`
Expected: FAIL — `LOCAL` not in enums, `_local_provider` attribute missing

- [ ] **Step 3: Add LOCAL to enums**

In `src/adam/llm/config.py:14-18`, add `LOCAL`:
```python
class ModelProvider(Enum):
    GROK = "grok"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL = "local"
```

In `src/adam/llm/async_client.py:68-73`, add `LOCAL`:
```python
class AsyncLLMProvider(Enum):
    """Available async providers"""
    XAI = "xai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL = "local"
```

- [ ] **Step 4: Add `_local_provider` attribute and update `_initialize_async_clients`**

In `src/adam/llm/async_client.py`, add to `__init__` (after `self._initialized = False`):
```python
        self._local_provider = None
```

At the end of `_initialize_async_clients()` (after the Gemini block, line ~163), add:
```python
        # Initialize local model provider
        local_enabled = os.getenv("LOCAL_MODEL_ENABLED", "true").lower() == "true"
        if local_enabled and OPENAI_AVAILABLE:
            try:
                from adam.llm.local_provider import LocalModelProvider
                self._local_provider = LocalModelProvider()
                await self._local_provider.discover()
                # Create an AsyncOpenAI client per discovered backend
                seen_urls = set()
                for model in self._local_provider.models.values():
                    if model.base_url not in seen_urls:
                        seen_urls.add(model.base_url)
                        self.clients[(AsyncLLMProvider.LOCAL, model.base_url)] = AsyncOpenAI(
                            base_url=model.base_url,
                            api_key="not-needed",
                            timeout=300.0,
                        )
                if self._local_provider.models:
                    self._client_locks[AsyncLLMProvider.LOCAL] = asyncio.Semaphore(3)
                    logger.info("Initialized LOCAL provider with %d model(s)", len(self._local_provider.models))
                    await self._local_provider.start_health_checks()
            except Exception as e:
                logger.error(f"Failed to initialize local provider: {e}")
                self._local_provider = LocalModelProvider(endpoints=[])
```

Add `import os` to the top of `_initialize_async_clients` if not already present.

- [ ] **Step 5: Update `_get_provider_for_model` to check local registry**

Replace the method at `src/adam/llm/async_client.py:532-544`:

```python
    def _get_provider_for_model(self, model: str) -> AsyncLLMProvider:
        """Get provider for a given model"""
        if model.startswith('grok-'):
            return AsyncLLMProvider.XAI
        elif model.startswith(('gpt-', 'o1-', 'o3')):
            return AsyncLLMProvider.OPENAI
        elif model.startswith('claude-'):
            return AsyncLLMProvider.ANTHROPIC
        elif model.startswith('gemini-'):
            return AsyncLLMProvider.GEMINI
        elif self._local_provider and self._local_provider.has_model(model):
            return AsyncLLMProvider.LOCAL
        else:
            # Default to XAI for unknown models (assuming they're Grok variants)
            return AsyncLLMProvider.XAI
```

- [ ] **Step 6: Add `_calculate_cost` zero-cost fallback for local models**

Find the existing `_calculate_cost` method in `async_client.py`. Add at the top of the method body:

```python
        # Local models are free
        if self._local_provider and self._local_provider.has_model(model):
            return 0.0
```

- [ ] **Step 7: Wire LOCAL provider into the completion flow**

In the `complete()` method (or `_complete_single`), where it selects the client by provider, add handling for LOCAL. The local client is stored as `(AsyncLLMProvider.LOCAL, base_url)` tuple key. Add before the existing provider dispatch:

```python
        if provider == AsyncLLMProvider.LOCAL:
            base_url = self._local_provider.get_base_url(model)
            client_key = (AsyncLLMProvider.LOCAL, base_url)
            client = self.clients.get(client_key)
            if client:
                # Reuse OpenAI-compatible completion path
                return await self._complete_openai(
                    client=client, prompt=prompt, model=model,
                    system_prompt=system_prompt, temperature=temperature,
                    max_tokens=max_tokens, stream=stream, **kwargs
                )
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /Users/vitoryago/ADAM && python -m pytest tests/test_local_integration.py -v`
Expected: All PASS

- [ ] **Step 9: Run full test suite for regressions**

Run: `cd /Users/vitoryago/ADAM && python -m pytest tests/ -v --timeout=30`
Expected: All existing tests still pass

- [ ] **Step 10: Commit**

```bash
git add src/adam/llm/async_client.py src/adam/llm/config.py tests/test_local_integration.py
git commit -m "feat: integrate LOCAL provider into AsyncLLMClient"
```

---

### Task 3: Local Models API Endpoint

**Files:**
- Create: `src/adam/api/routers/local_models.py`
- Modify: `src/adam/api/main.py:66-81` (register router)
- Test: `tests/test_local_models_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_local_models_api.py
"""Tests for GET /api/local-models endpoint."""
import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport
from adam.api.main import app
from adam.llm.local_provider import LocalModel


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
class TestLocalModelsAPI:
    async def test_list_local_models_returns_200(self, client):
        resp = await client.get("/api/local-models")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_list_local_models_returns_model_data(self, client):
        mock_models = [
            LocalModel(
                model_id="qwen3.5:72b-q4_K_M",
                display_name="Qwen3.5 72B (Q4_K_M)",
                backend="ollama",
                base_url="http://localhost:11434/v1",
                parameter_count=72,
                quantization="q4_K_M",
                available=True,
            ),
        ]
        with patch(
            "adam.api.routers.local_models.get_local_provider",
            return_value=MagicMock(get_available_models=MagicMock(return_value=mock_models)),
        ):
            resp = await client.get("/api/local-models")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) >= 1
            first = data[0]
            assert first["model_id"] == "qwen3.5:72b-q4_K_M"
            assert first["parameter_count"] == 72
            assert first["available"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/vitoryago/ADAM && python -m pytest tests/test_local_models_api.py -v`
Expected: FAIL — router does not exist

- [ ] **Step 3: Implement the router**

Create `src/adam/api/routers/local_models.py`:

```python
"""Local Models API — exposes discovered local models to frontends."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class LocalModelResponse(BaseModel):
    """Response schema for a local model entry."""
    model_id: str
    display_name: str
    backend: str
    parameter_count: int
    quantization: str
    available: bool


def get_local_provider():
    """Get the global LocalModelProvider instance from the LLM service.

    Returns a provider with an empty registry if local models are disabled
    or the service hasn't initialized yet.
    """
    try:
        from adam.services.llm_service import LLMService
        service = LLMService()
        if service.llm_client and service.llm_client._local_provider:
            return service.llm_client._local_provider
    except Exception:
        pass

    from adam.llm.local_provider import LocalModelProvider
    return LocalModelProvider(endpoints=[])


@router.get("", response_model=List[LocalModelResponse])
async def list_local_models():
    """Return all available local models."""
    provider = get_local_provider()
    available = provider.get_available_models()
    return [
        LocalModelResponse(
            model_id=m.model_id,
            display_name=m.display_name,
            backend=m.backend,
            parameter_count=m.parameter_count,
            quantization=m.quantization,
            available=m.available,
        )
        for m in available
    ]
```

- [ ] **Step 4: Register the router in main.py**

In `src/adam/api/main.py`, add `local_models` to the import block (line 66-70):

```python
from adam.api.routers import (
    conversations, projects, memories, messages,
    voice, voice_streaming, lineage, styles, dbt,
    deep_discussion, local_models,
)
```

Add the router include (after the `deep_discussion` line, around line 81):

```python
app.include_router(local_models.router, prefix="/api/local-models", tags=["local-models"])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/vitoryago/ADAM && python -m pytest tests/test_local_models_api.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/adam/api/routers/local_models.py src/adam/api/main.py tests/test_local_models_api.py
git commit -m "feat: add GET /api/local-models endpoint"
```

---

### Task 4: Deep Discussion — `prefer_local` in Config and Smart Defaults

**Files:**
- Modify: `src/adam/deep_discussion/config.py:18-32` (get_smart_defaults)
- Modify: `src/adam/api/models.py:249-261` (Pydantic schemas)
- Modify: `src/adam/api/routers/deep_discussion.py:37-70` (create_session)
- Test: `tests/test_local_prefer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_local_prefer.py
"""Tests for prefer_local in Deep Discussion config."""
import pytest
from unittest.mock import patch, MagicMock

from adam.deep_discussion.config import get_smart_defaults
from adam.api.models import DeepDiscussionSessionCreate
from adam.llm.local_provider import LocalModel


class TestSmartDefaultsPreferLocal:
    def test_prefer_local_false_returns_cloud_defaults(self):
        result = get_smart_defaults(prefer_local=False)
        assert result["reasoner"] == "grok-4.20-multi-agent-0309"
        assert result["coder"] == "claude-opus-4-6"

    def test_prefer_local_true_with_no_local_models_returns_cloud(self):
        """Falls back to cloud when no local models available."""
        with patch("adam.deep_discussion.config._get_local_models", return_value=[]):
            result = get_smart_defaults(prefer_local=True)
        assert result["reasoner"] == "grok-4.20-multi-agent-0309"

    def test_prefer_local_true_assigns_best_local_model(self):
        """Assigns the highest-parameter local model to all roles."""
        mock_models = [
            LocalModel(
                model_id="small:7b", display_name="Small 7B", backend="ollama",
                base_url="http://localhost:11434/v1", parameter_count=7,
                quantization="q4_K_M", available=True,
            ),
            LocalModel(
                model_id="qwen3.5:72b-q4_K_M", display_name="Qwen 3.5 72B",
                backend="ollama", base_url="http://localhost:11434/v1",
                parameter_count=72, quantization="q4_K_M", available=True,
            ),
        ]
        with patch("adam.deep_discussion.config._get_local_models", return_value=mock_models):
            result = get_smart_defaults(prefer_local=True)
        assert result["reasoner"] == "qwen3.5:72b-q4_K_M"
        assert result["coder"] == "qwen3.5:72b-q4_K_M"
        assert result["critic"] == "qwen3.5:72b-q4_K_M"
        assert result["synthesizer"] == "qwen3.5:72b-q4_K_M"

    def test_backward_compatible_no_args(self):
        """Calling with no args still works (existing code)."""
        result = get_smart_defaults()
        assert "reasoner" in result
        assert "coder" in result


class TestSessionCreateSchema:
    def test_prefer_local_defaults_false(self):
        create = DeepDiscussionSessionCreate(
            project_id="proj-1",
            question="test",
            pattern="peer_review",
        )
        assert create.prefer_local is False

    def test_prefer_local_can_be_set(self):
        create = DeepDiscussionSessionCreate(
            project_id="proj-1",
            question="test",
            pattern="peer_review",
            prefer_local=True,
        )
        assert create.prefer_local is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vitoryago/ADAM && python -m pytest tests/test_local_prefer.py -v`
Expected: FAIL — `prefer_local` param doesn't exist

- [ ] **Step 3: Update `get_smart_defaults` in config.py**

Replace `src/adam/deep_discussion/config.py` content:

```python
"""Session configuration for Deep Discussion Mode.

Provides ``SessionConfig`` (a dataclass with smart defaults) and helper
constants for the available models a user can choose from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local model bridge
# ---------------------------------------------------------------------------


def _get_local_models() -> list:
    """Return available local models, or empty list if provider not ready."""
    try:
        from adam.services.llm_service import LLMService
        service = LLMService()
        if service.llm_client and service.llm_client._local_provider:
            return service.llm_client._local_provider.get_available_models()
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Smart defaults
# ---------------------------------------------------------------------------


def get_smart_defaults(prefer_local: bool = False) -> Dict[str, str]:
    """Return the default model-per-role mapping.

    Args:
        prefer_local: When True, assign the most capable local model to
            all roles. Falls back to cloud defaults if no local models
            are available. User can override individual roles afterward.

    Returns:
        A dict with keys ``reasoner``, ``coder``, ``critic``, ``synthesizer``.
    """
    if prefer_local:
        local_models = _get_local_models()
        if local_models:
            best_local = max(local_models, key=lambda m: m.parameter_count)
            # Assign best local model to all roles as a starting point.
            # User can override individual roles to cloud via the config screen.
            return {
                "reasoner": best_local.model_id,
                "coder": best_local.model_id,
                "critic": best_local.model_id,
                "synthesizer": best_local.model_id,
            }

    return {
        "reasoner": "grok-4.20-multi-agent-0309",
        "coder": "claude-opus-4-6",
        "critic": "gpt-5.4-2026-03-05",
        "synthesizer": "claude-sonnet-4-6",
    }


# ---------------------------------------------------------------------------
# Available models catalogue
# ---------------------------------------------------------------------------

#: All models the user may assign to any role, keyed by human-readable label.
#: Grouped loosely by provider for display purposes.
AVAILABLE_MODELS: Dict[str, str] = {
    # Anthropic
    "Claude Opus 4.6": "claude-opus-4-6",
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
    "Claude Haiku 3.5": "claude-haiku-3-5",
    # OpenAI
    "GPT-5.4 (2026-03-05)": "gpt-5.4-2026-03-05",
    "GPT-4o": "gpt-4o",
    "o3": "o3",
    # xAI
    "Grok 4.20 Multi-Agent": "grok-4.20-multi-agent-0309",
    "Grok 3": "grok-3",
    # Google
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.0 Flash": "gemini-2.0-flash",
}


# ---------------------------------------------------------------------------
# SessionConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class SessionConfig:
    """Configuration for a single Deep Discussion session.

    Attributes:
        question: The user's question or topic to discuss.
        pattern: Orchestration pattern — one of ``"sequential"``,
            ``"debate"``, or ``"peer_review"``.
        model_assignments: Maps canonical role names to model IDs.
            Defaults to :func:`get_smart_defaults` when not provided.
        budget: Maximum spend in USD for the session. Defaults to ``2.0``.
        prefer_local: When True, smart defaults prefer local models.
        conversation_id: Optional ID of an existing ADAM conversation to
            associate this session with.
        conversation_context: Optional prior conversation text injected as
            extra context for all agents.
    """

    question: str
    pattern: str  # "sequential" | "debate" | "peer_review"
    model_assignments: Dict[str, str] = field(default_factory=dict)
    budget: float = 2.0
    prefer_local: bool = False
    conversation_id: Optional[str] = None
    conversation_context: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.model_assignments:
            self.model_assignments = get_smart_defaults(prefer_local=self.prefer_local)
```

- [ ] **Step 4: Add `prefer_local` to Pydantic schemas in models.py**

In `src/adam/api/models.py`, update `DeepDiscussionSessionCreate` (line ~249):

```python
class DeepDiscussionSessionCreate(BaseModel):
    """Request model for creating a deep discussion session"""
    project_id: str
    question: str
    pattern: str = "peer_review"
    conversation_id: Optional[str] = None
    prefer_local: bool = False
```

Add `prefer_local` column to `DeepDiscussionSessionDB` (after `budget` column, line ~230):

```python
    prefer_local = Column(Boolean, nullable=False, default=False)
```

Update `DeepDiscussionSessionDB.__init__` defaults:

```python
        kwargs.setdefault('prefer_local', False)
```

Add `prefer_local` to `DeepDiscussionSessionResponse`:

```python
    prefer_local: bool = False
```

- [ ] **Step 5: Pass `prefer_local` through the create_session endpoint**

In `src/adam/api/routers/deep_discussion.py`, update the `create_session` handler (line ~54-63). Change the `smart_defaults` call:

```python
    smart_defaults = get_smart_defaults(prefer_local=session_data.prefer_local)

    session = DeepDiscussionSessionDB(
        project_id=session_data.project_id,
        conversation_id=session_data.conversation_id,
        question=session_data.question,
        pattern=session_data.pattern,
        model_assignments=smart_defaults,
        prefer_local=session_data.prefer_local,
    )
```

- [ ] **Step 6: Add budget auto-adjustment when prefer_local is ON**

In `src/adam/api/routers/deep_discussion.py`, inside `create_session`, after computing `smart_defaults`, adjust the default budget when most roles are local:

```python
    # Auto-adjust budget when prefer_local assigns local models
    budget = 2.0
    if session_data.prefer_local:
        local_count = sum(
            1 for m in smart_defaults.values()
            if not any(m.startswith(p) for p in ("grok-", "claude-", "gpt-", "gemini-"))
        )
        if local_count >= 3:
            budget = 0.50

    session = DeepDiscussionSessionDB(
        ...
        budget=budget,
        ...
    )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /Users/vitoryago/ADAM && python -m pytest tests/test_local_prefer.py -v`
Expected: All PASS

- [ ] **Step 8: Run full test suite for regressions**

Run: `cd /Users/vitoryago/ADAM && python -m pytest tests/ -v --timeout=30`
Expected: All existing tests still pass

- [ ] **Step 8: Commit**

```bash
git add src/adam/deep_discussion/config.py src/adam/api/models.py src/adam/api/routers/deep_discussion.py tests/test_local_prefer.py
git commit -m "feat: add prefer_local to Deep Discussion config and smart defaults"
```

---

### Task 5: Update .env.example

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Add local model env vars**

Append to `.env.example`:

```
# Local Model Serving
LOCAL_MODEL_ENABLED=true
LOCAL_MODEL_ENDPOINTS=http://localhost:11434
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add LOCAL_MODEL_ENABLED and LOCAL_MODEL_ENDPOINTS to .env.example"
```

---

### Task 6: React Frontend — "Prefer Local" Toggle and Local Model Group

**Files:**
- Modify: `frontend/AdamChat/client/src/components/deep-discussion/model-selector.tsx`
- Modify: `frontend/AdamChat/client/src/components/deep-discussion/config-screen.tsx:58-65` (state + toggle)
- Modify: `frontend/AdamChat/client/src/lib/deep-discussion-api.ts:10-24` (pass prefer_local)

- [ ] **Step 1: Update ModelSelector to accept and render local models**

Replace `frontend/AdamChat/client/src/components/deep-discussion/model-selector.tsx`:

```tsx
import { cn } from "@/lib/utils";
import { useQuery } from "@tanstack/react-query";

interface LocalModel {
  model_id: string;
  display_name: string;
  backend: string;
  parameter_count: number;
  quantization: string;
  available: boolean;
}

interface ModelSelectorProps {
  value: string;
  onChange: (modelId: string) => void;
  className?: string;
}

const CLOUD_MODEL_GROUPS = [
  {
    label: "X.AI",
    models: [
      { id: "grok-4.20-multi-agent-0309", name: "Grok Multi-Agent" },
      { id: "grok-4.20-0309-reasoning", name: "Grok Reasoning" },
      { id: "grok-4.20-0309-non-reasoning", name: "Grok Standard" },
    ],
  },
  {
    label: "Anthropic",
    models: [
      { id: "claude-opus-4-6", name: "Claude Opus" },
      { id: "claude-sonnet-4-6", name: "Claude Sonnet" },
      { id: "claude-haiku-4-5", name: "Claude Haiku" },
    ],
  },
  {
    label: "OpenAI",
    models: [
      { id: "gpt-5.4-2026-03-05", name: "GPT-5.4" },
      { id: "gpt-5.4-mini-2026-03-17", name: "GPT-5.4 Mini" },
    ],
  },
  {
    label: "Google",
    models: [
      { id: "gemini-3.1-pro-preview", name: "Gemini Pro" },
      { id: "gemini-3-flash-preview", name: "Gemini Flash" },
    ],
  },
];

export function ModelSelector({ value, onChange, className }: ModelSelectorProps) {
  const { data: localModels = [] } = useQuery<LocalModel[]>({
    queryKey: ["/api/local-models"],
    queryFn: async () => {
      const res = await fetch("/api/local-models");
      if (!res.ok) return [];
      return res.json();
    },
    staleTime: 30_000,
  });

  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={cn(
        "w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
        "focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-1",
        "cursor-pointer",
        className,
      )}
    >
      {localModels.length > 0 && (
        <optgroup label="Local  $0.00">
          {localModels.map((m) => (
            <option key={m.model_id} value={m.model_id}>
              {m.display_name}
            </option>
          ))}
        </optgroup>
      )}
      {CLOUD_MODEL_GROUPS.map((group) => (
        <optgroup key={group.label} label={group.label}>
          {group.models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  );
}
```

- [ ] **Step 2: Add "Prefer Local" toggle to ConfigScreen**

In `frontend/AdamChat/client/src/components/deep-discussion/config-screen.tsx`, add state for `preferLocal` (after line 64, the `budget` state):

```tsx
  const [preferLocal, setPreferLocal] = useState(false);
```

Add the toggle UI inside the config form, above the agent cards grid. Find the section where budget/advanced settings render and add before the agent cards:

```tsx
          <div className="flex items-center justify-between py-2">
            <label className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              <span>Prefer Local Models</span>
            </label>
            <button
              type="button"
              role="switch"
              aria-checked={preferLocal}
              onClick={() => setPreferLocal(!preferLocal)}
              className={cn(
                "relative inline-flex h-5 w-9 items-center rounded-full transition-colors",
                preferLocal ? "bg-primary" : "bg-muted",
              )}
            >
              <span
                className={cn(
                  "inline-block h-3.5 w-3.5 transform rounded-full bg-background transition-transform",
                  preferLocal ? "translate-x-4" : "translate-x-0.5",
                )}
              />
            </button>
          </div>
```

- [ ] **Step 3: Pass `prefer_local` through the API call**

In `frontend/AdamChat/client/src/lib/deep-discussion-api.ts`, update `createSession` to accept and pass `prefer_local`:

```typescript
export async function createSession(
  projectId: string,
  question: string,
  pattern: string,
  conversationId?: string,
  preferLocal?: boolean,
) {
  const res = await fetch(`${API_BASE}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      project_id: projectId,
      question,
      pattern,
      conversation_id: conversationId,
      prefer_local: preferLocal ?? false,
    }),
    credentials: "include",
  });
  await throwIfResNotOk(res);
  return res.json();
}
```

Update the `startMutation` call in `config-screen.tsx` to pass `preferLocal` to `createSession`.

- [ ] **Step 4: Verify TypeScript compilation**

Run: `cd /Users/vitoryago/ADAM/frontend/AdamChat && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add frontend/AdamChat/client/src/components/deep-discussion/model-selector.tsx \
       frontend/AdamChat/client/src/components/deep-discussion/config-screen.tsx \
       frontend/AdamChat/client/src/lib/deep-discussion-api.ts
git commit -m "feat: add Prefer Local toggle and local model group to React config"
```

---

### Task 7: VS Code Extension — "Prefer Local" Toggle and Local Model Group

**Files:**
- Modify: `vscode-extension/adam-code/media/deep-discussion.js:22-48` (MODEL_GROUPS + state)
- Modify: `vscode-extension/adam-code/media/deep-discussion.js:142-316` (renderConfig)
- Modify: `vscode-extension/adam-code/media/deep-discussion.js:582-625` (handleStart)
- Modify: `vscode-extension/adam-code/src/providers/deepDiscussionProvider.ts:316-324` (handleStartSession)
- Modify: `vscode-extension/adam-code/src/client/adamClient.ts` (pass prefer_local)

- [ ] **Step 1: Add `preferLocal` state and fetch local models**

In `vscode-extension/adam-code/media/deep-discussion.js`, add after `var advancedExpanded = false;` (line 17):

```javascript
    var preferLocal = false;
    var localModels = [];  // fetched from /api/local-models
```

Add a function to fetch local models (after the `escapeHtml` function):

```javascript
    function fetchLocalModels() {
        vscode.postMessage({ type: 'fetchLocalModels' });
    }
```

- [ ] **Step 2: Add local models to renderConfig dropdown**

In the `renderConfig` function, add the "Prefer Local" toggle before the model dropdowns (after the `advToggle` block, before the `advSection` model selects):

```javascript
        // Prefer Local toggle
        var localRow = document.createElement('div');
        localRow.className = 'agent-config-row';
        var localLabel = document.createElement('label');
        localLabel.textContent = 'Prefer Local';
        localRow.appendChild(localLabel);
        var localToggle = document.createElement('button');
        localToggle.className = preferLocal ? 'toggle-btn on' : 'toggle-btn';
        localToggle.textContent = preferLocal ? 'ON' : 'OFF';
        localToggle.addEventListener('click', function() {
            preferLocal = !preferLocal;
            renderConfig(prefill);
        });
        localRow.appendChild(localToggle);
        advSection.appendChild(localRow);
```

Update the model `<select>` builder to include local models at the top:

```javascript
            // Add local models group if available
            if (localModels.length > 0) {
                var localGroup = document.createElement('optgroup');
                localGroup.label = 'Local  $0.00';
                for (var lm = 0; lm < localModels.length; lm++) {
                    var lmOpt = document.createElement('option');
                    lmOpt.value = localModels[lm].model_id;
                    lmOpt.textContent = localModels[lm].display_name;
                    if (localModels[lm].model_id === defaultModel) { lmOpt.selected = true; }
                    localGroup.appendChild(lmOpt);
                }
                sel.appendChild(localGroup);
            }
```

- [ ] **Step 3: Pass `preferLocal` in handleStart**

In the `handleStart` function, add `preferLocal` to the message:

```javascript
        vscode.postMessage({
            type: 'startSession',
            question: question,
            pattern: pattern,
            modelAssignments: modelAssignments,
            budget: budget,
            preferLocal: preferLocal
        });
```

- [ ] **Step 4: Handle `fetchLocalModels` and pass `preferLocal` in provider**

In `vscode-extension/adam-code/src/providers/deepDiscussionProvider.ts`, add a case for `fetchLocalModels` in the message handler:

```typescript
                case 'fetchLocalModels':
                    await this.handleFetchLocalModels();
                    break;
```

Add the handler method:

```typescript
    private async handleFetchLocalModels() {
        try {
            const response = await fetch(`${this.adamClient.baseURL}/api/local-models`);
            if (response.ok) {
                const models = await response.json();
                this._view?.webview.postMessage({ type: 'localModelsData', models });
            }
        } catch (_) {
            // Silently fail — local models are optional
        }
    }
```

Update `handleStartSession` signature to accept `preferLocal`:

```typescript
    private async handleStartSession(
        question: string,
        pattern: string,
        modelAssignments?: Record<string, string>,
        budget?: number,
        preferLocal?: boolean
    ) {
```

Pass `preferLocal` when creating the session (update the `createDeepDiscussionSession` call).

- [ ] **Step 5: Handle `localModelsData` message in webview JS**

In `deep-discussion.js`, add a case in the message handler:

```javascript
            case 'localModelsData':
                localModels = msg.models || [];
                if (mode === 'config') { renderConfig(); }
                break;
```

- [ ] **Step 6: Add toggle button CSS to deep-discussion.css**

Append to `vscode-extension/adam-code/media/deep-discussion.css`:

```css
/* Toggle button for Prefer Local */
.toggle-btn {
    padding: 2px 10px;
    border-radius: 10px;
    border: 1px solid var(--vscode-panel-border);
    background-color: var(--vscode-input-background);
    color: var(--vscode-descriptionForeground);
    font-size: 11px;
    cursor: pointer;
    transition: background-color 0.2s, color 0.2s;
}

.toggle-btn.on {
    background-color: var(--vscode-button-background);
    color: var(--vscode-button-foreground);
    border-color: var(--vscode-button-background);
}
```

- [ ] **Step 7: Fetch local models on init**

In `deep-discussion.js`, update the initial setup (line ~829) to also fetch local models:

```javascript
    vscode.postMessage({ type: 'loadHistory' });
    fetchLocalModels();
    render();
```

- [ ] **Step 8: Verify compilation**

Run: `cd /Users/vitoryago/ADAM/vscode-extension/adam-code && npm run compile`
Expected: No errors

- [ ] **Step 9: Commit**

```bash
git add vscode-extension/adam-code/media/deep-discussion.js \
       vscode-extension/adam-code/media/deep-discussion.css \
       vscode-extension/adam-code/src/providers/deepDiscussionProvider.ts \
       vscode-extension/adam-code/src/client/adamClient.ts
git commit -m "feat: add Prefer Local toggle and local model group to VS Code extension"
```

---

### Task 8: Update Deep Discussion Notebook

**Files:**
- Modify: `deep_discussion.md`

- [ ] **Step 1: Add local model integration to the notebook**

Add a new section after "## What's Next" and update the status table. Add under Key Decisions:

```markdown
| Local model support | Generic LocalModelProvider (OpenAI-compatible) | Backend-agnostic, future-proof for LoRA fine-tuning |
| Local model UX | "Prefer Local" toggle in Advanced settings | Clean, non-intrusive — cloud-only by default |
```

Update the "What's Next" table to mark "Custom/local model providers" as Done and add LoRA fine-tuning as the next milestone.

- [ ] **Step 2: Commit**

```bash
git add deep_discussion.md
git commit -m "docs: add local model integration to project notebook"
```
