# Local Model Integration — Design Spec

**Date:** 2026-03-31
**Status:** Draft
**Depends on:** Deep Discussion Mode (complete), Gemini provider integration (complete)

## Context

ADAM's Deep Discussion Mode calls cloud APIs (Grok, Claude, OpenAI, Gemini) for every agent role. Each session costs $0.50–$2.00. The roadmap includes custom/local model providers (Medium priority) and LoRA fine-tuning of frontier OSS models (future).

This spec adds a generic local model integration layer so ADAM can discover, route to, and use any OpenAI-compatible local inference server (Ollama, vLLM, MLX Server, LM Studio) alongside cloud providers. This is the foundation that fine-tuned quantized models will plug into.

### Motivation

- **Cost:** Local inference is $0.00/token. A session mixing local Critic + cloud Coder drops cost from ~$1.50 to ~$0.40.
- **Sovereignty:** Models run on the user's hardware. No data leaves the machine for local calls.
- **Future-proofing:** TurboQuant-style extreme quantization (3-4 bit, zero accuracy loss) makes 70B models runnable on Apple Silicon (M4 Pro 48GB). As quantization improves, more frontier models become local-viable.
- **LoRA pipeline:** Fine-tuned specialist models (starting with Critic) need local serving infrastructure to plug into.

### Target Hardware

M4 Pro with 48GB unified memory. Realistic model ceiling:

| Model Class | 4-bit VRAM | Fits? | Speed |
|------------|-----------|-------|-------|
| 7–13B | ~4–7 GB | Easily | Fast (50+ tok/s) |
| 30–40B | ~15–20 GB | Comfortably | Good (15–25 tok/s) |
| 70B | ~35 GB | Tight but yes | Usable (8–12 tok/s) |
| 120B+ | ~60 GB | No | — |

Sweet spot: 30–70B quantized models. Candidate base models: Qwen 3.5, GLM 5.1, DeepSeek Coder, Kimi 2.5.

---

## Scope

### In Scope

- `LocalModelProvider` class: discovery, health checks, model registry
- `LOCAL` provider in `AsyncLLMClient` using OpenAI-compatible API
- Auto-detection of Ollama, vLLM, MLX Server, LM Studio by probing known ports
- Dynamic model discovery via `/v1/models` endpoint
- `GET /api/local-models` endpoint for frontend to query available models
- "Prefer Local Models" toggle in Deep Discussion config (React + VS Code)
- Local models in model selector dropdowns (Local optgroup, `🖥️` icon, `$0.00`)
- Smart defaults that assign local models to roles when toggle is ON
- Budget auto-adjustment when local models are assigned
- Zero-cost tracking for local model calls in PricingManager
- Background health checks (30s interval)

### Out of Scope

- Model management (pull, delete, quantize) — user manages via Ollama CLI / vLLM / etc.
- LoRA fine-tuning pipeline — separate spec
- Training data generation — separate spec
- TurboQuant implementation — ADAM consumes quantized models, doesn't quantize them
- Automatic capability ranking / benchmarking of local models
- Local models in regular chat (non–Deep Discussion) — deferred

---

## Architecture

### New Module: `src/adam/llm/local_provider.py`

**`LocalModelProvider`** (class):

Abstracts any OpenAI-compatible local inference server. Backend-agnostic — works with anything that serves `/v1/chat/completions`.

**Discovery:** Probes a configurable list of endpoints on startup and on-demand:

| Backend | Default Port | Probe URL |
|---------|-------------|-----------|
| Ollama | 11434 | `GET /api/tags` + `GET /v1/models` |
| vLLM | 8000 | `GET /v1/models` |
| MLX Server | 5000 | `GET /v1/models` |
| LM Studio | 1234 | `GET /v1/models` |
| Custom | configurable | `GET /v1/models` |

Each probe has a 2-second timeout. All probes run in parallel. If no endpoints are reachable, ADAM proceeds cloud-only — this is not an error.

**Model registry:** After discovery, builds `Dict[str, LocalModel]`:

```python
@dataclass
class LocalModel:
    model_id: str           # e.g., "qwen3.5:72b-q4_K_M"
    display_name: str       # e.g., "Qwen 3.5 72B (Q4)"
    backend: str            # "ollama" | "vllm" | "mlx" | "lmstudio" | "custom"
    base_url: str           # e.g., "http://localhost:11434/v1"
    parameter_count: int    # extracted from name/metadata, in billions
    quantization: str       # "Q4_K_M", "Q3_K_S", "FP16", etc.
    available: bool         # current health status
```

**Health checks:** Background asyncio task pings each backend every 30 seconds via `GET /v1/models`. If unreachable, marks models unavailable. If it recovers, re-discovers and marks available. Logs transitions.

**No model management:** ADAM does not pull, delete, or configure models. It only discovers what's already running.

---

### AsyncLLMClient Integration

**New provider:** Add `LOCAL = "local"` to `AsyncLLMProvider` enum.

**Model routing in `_get_provider_for_model()`:**

```
1. Fixed prefix checks (existing):
   - grok-*    → XAI
   - claude-*  → ANTHROPIC
   - gpt-*     → OPENAI
   - gemini-*  → GEMINI
2. NEW — Local registry check:
   - if model_id in LocalModelProvider.models → LOCAL
3. Fallback: raise unknown model error
```

Order matters: cloud prefixes take priority. A local model named `gpt-something` would still route to OpenAI. Local model IDs come from the inference server and are typically namespaced (e.g., `qwen3.5:72b-q4_K_M`, `deepseek-coder-v3:33b`).

**Client initialization:** In `_initialize_async_clients()`, if `LOCAL_MODEL_ENABLED=true`:
- Instantiate `LocalModelProvider`, run initial discovery
- Create one `AsyncOpenAI(base_url=...)` client per discovered backend
- Start background health check task

**Inference:** Reuse existing `_complete_openai` / `_stream_openai` methods unchanged. Local servers speak the same OpenAI chat completions API.

**Cost tracking:** `PricingManager` gets a fallback: if model not in pricing table, check `LocalModelProvider.models` — if present, return zero cost. CostGuard still checks budget between steps, but local calls don't consume it.

**No changes to:** Scratchpad, CostGuard, patterns, orchestrator, or any existing provider.

---

### Deep Discussion UX: "Prefer Local" Toggle

**React config screen (`config-screen.tsx`):** New toggle in Advanced Settings, above the model dropdowns:

```
▸ Advanced Configuration
┌──────────────────────────────────┐
│ ⚡ Prefer Local Models    [ON]   │
│                                  │
│ Reasoner: [Qwen 3.5 72B 🖥️  ▼] │
│ Coder:    [Claude Opus ☁️    ▼] │
│ Critic:   [Qwen 3.5 72B 🖥️  ▼] │
│ Synth:    [Qwen 3.5 72B 🖥️  ▼] │
│ Budget:   [$0.50 ═══]           │
└──────────────────────────────────┘
```

**Toggle behavior:**
- **ON:** `get_smart_defaults(prefer_local=True)` queries `LocalModelProvider` for available models. Assigns the most capable local model (by parameter count) to each role. Falls back to cloud for roles where no suitable local model is available.
- **OFF:** Current cloud-only defaults (Grok MA → Reasoner, Opus → Coder, GPT-5.4 → Critic, Sonnet → Synth).
- **Default state:** OFF if no local models detected, ON if at least one local model is available.

**Model selector dropdown:** Two `<optgroup>` sections:
- "Local" group at top: models with `🖥️` icon and `$0.00`
- Cloud provider groups below: models with `☁️` icon and price estimate

**Budget auto-adjustment:** When "Prefer Local" is ON and 3+ roles are local, default budget drops from $2.00 to $0.50 (only remaining cloud calls cost money). User can still override.

**VS Code extension:** Same toggle in Advanced section of `deep-discussion.js`. The provider passes `prefer_local` to the API.

---

### API Changes

**`get_smart_defaults()` update:**

```python
def get_smart_defaults(prefer_local: bool = False) -> dict:
    """Return default model assignments. If prefer_local, use local models where available."""
    if prefer_local:
        local_models = LocalModelProvider.get_available_models()
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
    # Cloud defaults (existing)
    return {
        "reasoner": "grok-4.20-multi-agent-0309",
        "coder": "claude-opus-4-6",
        "critic": "gpt-5.4-2026-03-05",
        "synthesizer": "claude-sonnet-4-6",
    }
```

**New endpoint:**

`GET /api/local-models` — returns list of discovered local models with metadata. Called by frontend to populate the Local optgroup.

```json
[
  {
    "model_id": "qwen3.5:72b-q4_K_M",
    "display_name": "Qwen 3.5 72B (Q4)",
    "backend": "ollama",
    "parameter_count": 72,
    "quantization": "Q4_K_M",
    "available": true
  }
]
```

**Session schema update:** Add `prefer_local: bool = False` to `DeepDiscussionSessionCreate` and `DeepDiscussionSessionDB`. The `POST /sessions` endpoint passes it to `get_smart_defaults()`.

---

### Configuration

**New environment variables:**

```
LOCAL_MODEL_ENABLED=true
LOCAL_MODEL_ENDPOINTS=http://localhost:11434,http://localhost:8000
```

- `LOCAL_MODEL_ENABLED` — master switch. Default `true`. Set `false` to skip all local model discovery.
- `LOCAL_MODEL_ENDPOINTS` — comma-separated list of base URLs to probe. Default: `http://localhost:11434` (Ollama). Add more as needed.

Added to `.env.example`.

---

## Files Summary

### New Files

| File | Purpose |
|------|---------|
| `src/adam/llm/local_provider.py` | `LocalModelProvider` — discovery, health, registry |
| `src/adam/api/routers/local_models.py` | `GET /api/local-models` endpoint |
| `tests/test_local_provider.py` | Discovery, health checks, model parsing, fallback |
| `tests/test_local_integration.py` | Routing, zero-cost tracking, prefer_local defaults |

### Modified Files

| File | Change |
|------|--------|
| `src/adam/llm/async_client.py` | Add `LOCAL` provider, registry-based routing |
| `src/adam/llm/config.py` | Add `LOCAL` to `ModelProvider`, zero-cost fallback |
| `src/adam/deep_discussion/config.py` | `get_smart_defaults(prefer_local)`, local models in `AVAILABLE_MODELS` |
| `src/adam/api/routers/deep_discussion.py` | Pass `prefer_local` through session create/update |
| `src/adam/api/models.py` | Add `prefer_local` bool to session schemas |
| `src/adam/api/main.py` | Register `local_models` router, start discovery in lifespan |
| `.env.example` | Add `LOCAL_MODEL_ENDPOINTS`, `LOCAL_MODEL_ENABLED` |
| `frontend/.../config-screen.tsx` | "Prefer Local" toggle, Local optgroup in dropdowns |
| `frontend/.../deep-discussion-api.ts` | Pass `prefer_local` field |
| `vscode/.../deep-discussion.js` | "Prefer Local" toggle, Local model group |
| `vscode/.../adamClient.ts` | Pass `prefer_local` field |

### Not Changed

Scratchpad, CostGuard, patterns (sequential/debate/peer_review), orchestrator, existing cloud providers, existing tests.

---

## Testing Strategy

- **Unit tests:** LocalModelProvider discovery with mocked HTTP responses, model ID parsing, health check transitions, empty-state fallback
- **Integration tests:** AsyncLLMClient routing to LOCAL provider, zero-cost tracking in PricingManager, `get_smart_defaults(prefer_local=True)` with mocked local registry
- **API tests:** `GET /api/local-models`, session creation with `prefer_local=true`, model assignment validation
- **Frontend:** TypeScript compilation check. Manual test of toggle + dropdown behavior.

---

## Design Principles

1. **Additive only** — zero changes to existing providers, patterns, or Scratchpad
2. **Backend-agnostic** — works with anything that speaks OpenAI-compatible API
3. **Graceful degradation** — no local models? Cloud-only, silently. Model dies mid-session? Existing `agent_error` handling kicks in.
4. **No model management** — ADAM discovers, it doesn't manage. User owns their model stack.
5. **Foundation for fine-tuning** — a LoRA-trained Critic served via Ollama slots in with zero additional ADAM changes
