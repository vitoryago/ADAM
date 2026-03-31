"""Local Model Provider -- discovers and manages locally-served LLMs.

Probes OpenAI-compatible endpoints (Ollama, vLLM, MLX Server, LM Studio)
to build a registry of available models. Backend-agnostic -- works with
anything that serves ``/v1/chat/completions``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINTS = ["http://localhost:11434"]
_PARAM_RE = re.compile(r"(\d+)[bB]")
_QUANT_RE = re.compile(r"(q\d+[_a-zA-Z]*)", re.IGNORECASE)


@dataclass
class LocalModel:
    """A single model served by a local inference backend."""

    model_id: str
    display_name: str
    backend: str
    base_url: str
    parameter_count: int
    quantization: str
    available: bool = True


class LocalModelProvider:
    """Discovers and tracks locally-served LLMs."""

    def __init__(self, endpoints: Optional[List[str]] = None) -> None:
        if endpoints is None:
            raw = os.getenv("LOCAL_MODEL_ENDPOINTS", "")
            endpoints = (
                [e.strip() for e in raw.split(",") if e.strip()]
                if raw
                else _DEFAULT_ENDPOINTS
            )
        self.endpoints = endpoints
        self.models: Dict[str, LocalModel] = {}
        self._health_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def discover(self) -> None:
        """Probe all endpoints and rebuild the model registry."""
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
            logger.info("No local models discovered -- proceeding cloud-only")

    async def _probe_endpoint(self, base_url: str) -> Dict[str, LocalModel]:
        """Query a single endpoint's /v1/models and return parsed models."""
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

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_model_entry(self, model_id: str, base_url: str) -> LocalModel:
        """Extract parameter count, quantization, and display name from a
        model identifier string (e.g. ``qwen3.5:72b-q4_K_M``)."""
        param_match = _PARAM_RE.search(model_id)
        parameter_count = int(param_match.group(1)) if param_match else 0

        quant_match = _QUANT_RE.search(model_id)
        quantization = quant_match.group(1) if quant_match else ""

        display_name = self._build_display_name(
            model_id, parameter_count, quantization
        )
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
        """Produce a human-friendly name like ``qwen3.5 72B (Q4_K_M)``."""
        name = model_id.split(":")[0] if ":" in model_id else model_id
        parts = [name]
        if param_count:
            parts.append(f"{param_count}B")
        if quant:
            parts.append(f"({quant.upper()})")
        return " ".join(parts)

    @staticmethod
    def _detect_backend(base_url: str) -> str:
        """Guess the backend type from the port number."""
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
    # Query helpers
    # ------------------------------------------------------------------

    def get_available_models(self) -> List[LocalModel]:
        """Return only models currently marked as available."""
        return [m for m in self.models.values() if m.available]

    def has_model(self, model_id: str) -> bool:
        """Check whether *model_id* exists and is available."""
        model = self.models.get(model_id)
        return model is not None and model.available

    def get_base_url(self, model_id: str) -> Optional[str]:
        """Return the base URL for *model_id*, or ``None``."""
        model = self.models.get(model_id)
        return model.base_url if model else None

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def start_health_checks(self, interval: float = 30.0) -> None:
        """Launch a background task that re-discovers models periodically."""
        if self._health_task is not None:
            return
        self._health_task = asyncio.create_task(self._health_loop(interval))

    async def stop_health_checks(self) -> None:
        """Cancel the background health-check loop."""
        if self._health_task is not None:
            self._health_task.cancel()
            self._health_task = None

    async def _health_loop(self, interval: float) -> None:
        """Continuously re-discover models at *interval* seconds."""
        while True:
            await asyncio.sleep(interval)
            try:
                await self.discover()
            except Exception as exc:
                logger.warning("Health check error: %s", exc)
