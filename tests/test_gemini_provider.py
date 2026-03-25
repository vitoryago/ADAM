"""
Tests for Gemini provider addition and updated model roster.

TDD test file — these tests define the expected state after Task 1 implementation.
"""


# ---------------------------------------------------------------------------
# ModelProvider enum
# ---------------------------------------------------------------------------

def test_model_provider_has_gemini():
    """ModelProvider enum must include GEMINI."""
    from adam.llm.config import ModelProvider
    assert hasattr(ModelProvider, "GEMINI")
    assert ModelProvider.GEMINI.value == "gemini"


def test_model_provider_has_all_four_providers():
    """All four providers must be present."""
    from adam.llm.config import ModelProvider
    provider_values = {p.value for p in ModelProvider}
    assert "grok" in provider_values
    assert "openai" in provider_values
    assert "anthropic" in provider_values
    assert "gemini" in provider_values


# ---------------------------------------------------------------------------
# LLMConfig — API key slot
# ---------------------------------------------------------------------------

def test_llmconfig_has_gemini_api_key_slot():
    """LLMConfig.api_keys must have a slot for ModelProvider.GEMINI."""
    from adam.llm.config import LLMConfig, ModelProvider
    cfg = LLMConfig()
    assert ModelProvider.GEMINI in cfg.api_keys


# ---------------------------------------------------------------------------
# LLMConfig — new model entries
# ---------------------------------------------------------------------------

def test_llmconfig_has_new_grok_models():
    """LLMConfig must contain the three new Grok model entries."""
    from adam.llm.config import LLMConfig
    cfg = LLMConfig()
    assert "grok-4.20-multi-agent-0309" in cfg.models
    assert "grok-4.20-0309-reasoning" in cfg.models
    assert "grok-4.20-0309-non-reasoning" in cfg.models


def test_llmconfig_has_new_claude_models():
    """LLMConfig must contain the three new Claude model entries."""
    from adam.llm.config import LLMConfig
    cfg = LLMConfig()
    assert "claude-opus-4-6" in cfg.models
    assert "claude-sonnet-4-6" in cfg.models
    assert "claude-haiku-4-5" in cfg.models


def test_llmconfig_has_new_openai_models():
    """LLMConfig must contain the two new OpenAI model entries."""
    from adam.llm.config import LLMConfig
    cfg = LLMConfig()
    assert "gpt-5.4-2026-03-05" in cfg.models
    assert "gpt-5.4-mini-2026-03-17" in cfg.models


def test_llmconfig_has_gemini_models():
    """LLMConfig must contain the two Gemini model entries."""
    from adam.llm.config import LLMConfig, ModelProvider
    cfg = LLMConfig()
    assert "gemini-3.1-pro-preview" in cfg.models
    assert "gemini-3-flash-preview" in cfg.models
    # Confirm they are assigned to the GEMINI provider
    assert cfg.models["gemini-3.1-pro-preview"].provider == ModelProvider.GEMINI
    assert cfg.models["gemini-3-flash-preview"].provider == ModelProvider.GEMINI


# ---------------------------------------------------------------------------
# LLMConfig — default_models updated
# ---------------------------------------------------------------------------

def test_llmconfig_default_models_use_new_names():
    """default_models must reference the new model names."""
    from adam.llm.config import LLMConfig
    cfg = LLMConfig()
    new_names = {
        "grok-4.20-multi-agent-0309",
        "grok-4.20-0309-reasoning",
        "grok-4.20-0309-non-reasoning",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "gpt-5.4-2026-03-05",
        "gpt-5.4-mini-2026-03-17",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
    }
    for key, model_name in cfg.default_models.items():
        assert model_name in new_names or model_name in cfg.models, (
            f"default_models['{key}'] = '{model_name}' not in new model roster"
        )


# ---------------------------------------------------------------------------
# AsyncLLMProvider enum
# ---------------------------------------------------------------------------

def test_async_provider_has_gemini():
    """AsyncLLMProvider enum must include GEMINI."""
    from adam.llm.async_client import AsyncLLMProvider
    assert hasattr(AsyncLLMProvider, "GEMINI")
    assert AsyncLLMProvider.GEMINI.value == "gemini"


# ---------------------------------------------------------------------------
# _get_provider_for_model routing
# ---------------------------------------------------------------------------

def test_gemini_model_routes_to_gemini_provider():
    """Models starting with 'gemini-' must route to AsyncLLMProvider.GEMINI."""
    from adam.llm.async_client import AsyncLLMClient, AsyncLLMProvider
    client = AsyncLLMClient.__new__(AsyncLLMClient)
    assert client._get_provider_for_model("gemini-3.1-pro-preview") == AsyncLLMProvider.GEMINI
    assert client._get_provider_for_model("gemini-3-flash-preview") == AsyncLLMProvider.GEMINI


def test_grok_model_still_routes_to_xai():
    """Models starting with 'grok-' must still route to AsyncLLMProvider.XAI."""
    from adam.llm.async_client import AsyncLLMClient, AsyncLLMProvider
    client = AsyncLLMClient.__new__(AsyncLLMClient)
    assert client._get_provider_for_model("grok-4.20-0309-reasoning") == AsyncLLMProvider.XAI
    assert client._get_provider_for_model("grok-4.20-non-reasoning") == AsyncLLMProvider.XAI


def test_claude_model_still_routes_to_anthropic():
    """Models starting with 'claude-' must still route to AsyncLLMProvider.ANTHROPIC."""
    from adam.llm.async_client import AsyncLLMClient, AsyncLLMProvider
    client = AsyncLLMClient.__new__(AsyncLLMClient)
    assert client._get_provider_for_model("claude-opus-4-6") == AsyncLLMProvider.ANTHROPIC
    assert client._get_provider_for_model("claude-haiku-4-5") == AsyncLLMProvider.ANTHROPIC


def test_gpt_model_still_routes_to_openai():
    """Models starting with 'gpt-' must still route to AsyncLLMProvider.OPENAI."""
    from adam.llm.async_client import AsyncLLMClient, AsyncLLMProvider
    client = AsyncLLMClient.__new__(AsyncLLMClient)
    assert client._get_provider_for_model("gpt-5.4-2026-03-05") == AsyncLLMProvider.OPENAI
    assert client._get_provider_for_model("gpt-5.4-mini-2026-03-17") == AsyncLLMProvider.OPENAI


# ---------------------------------------------------------------------------
# ModelTier values updated
# ---------------------------------------------------------------------------

def test_model_tier_reasoning_updated():
    """ModelTier.REASONING must use the new grok-4.20 reasoning model."""
    from adam.llm.router import ModelTier
    assert ModelTier.REASONING.value == "grok-4.20-0309-reasoning"


def test_model_tier_standard_updated():
    """ModelTier.STANDARD must use the new grok-4.20 non-reasoning model."""
    from adam.llm.router import ModelTier
    assert ModelTier.STANDARD.value == "grok-4.20-0309-non-reasoning"


def test_model_tier_fast_updated():
    """ModelTier.FAST must use the new grok-4.20 non-reasoning model."""
    from adam.llm.router import ModelTier
    assert ModelTier.FAST.value == "grok-4.20-0309-non-reasoning"


def test_model_tier_router_updated():
    """ModelTier.ROUTER must use claude-haiku-4-5."""
    from adam.llm.router import ModelTier
    assert ModelTier.ROUTER.value == "claude-haiku-4-5"
