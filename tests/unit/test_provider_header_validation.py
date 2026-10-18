import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import astrbot.dashboard.services.config_service as config_service_module
from astrbot.core.provider.headers import HeaderTemplateError
from astrbot.core.provider.manager import ProviderManager
from astrbot.dashboard.services.config_service import ProviderConfigService


class _FakeConfig(dict):
    def __init__(self, data: dict) -> None:
        super().__init__(copy.deepcopy(data))
        self.saved = 0

    def save_config(self, *_args, **_kwargs) -> None:
        self.saved += 1


class _FakeConfigManager:
    def __init__(self, config: _FakeConfig) -> None:
        self.confs = {"default": config}
        self.default_conf = config


class _FakePersonaManager:
    default_persona = "default"


def _make_provider_manager(
    provider_configs: list[dict],
    provider_sources: list[dict] | None = None,
) -> tuple[ProviderManager, _FakeConfig]:
    config = _FakeConfig(
        {
            "provider": provider_configs,
            "provider_sources": provider_sources or [],
            "provider_settings": {},
        }
    )
    manager = ProviderManager(
        _FakeConfigManager(config),
        db_helper=None,
        persona_mgr=_FakePersonaManager(),
    )
    return manager, config


INVALID_TEMPLATE_HEADERS = {"X-Bad": "{unknown}"}


@pytest.mark.asyncio
async def test_create_provider_rejects_invalid_header_template(monkeypatch):
    manager, config = _make_provider_manager([])

    async def fail_load(_provider_config):
        pytest.fail("provider must not load when validation fails")

    monkeypatch.setattr(manager, "load_provider", fail_load)

    with pytest.raises(HeaderTemplateError):
        await manager.create_provider(
            {
                "id": "p1",
                "type": "openai_chat_completion",
                "provider_type": "chat_completion",
                "model": "gpt-4o-mini",
                "custom_headers": dict(INVALID_TEMPLATE_HEADERS),
            }
        )

    assert config["provider"] == []
    assert config.saved == 0


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_create_provider_without_provider_type_checks_syntax_only(monkeypatch):
    manager, config = _make_provider_manager([])

    loaded: list[dict] = []

    async def fake_load(provider_config):
        loaded.append(provider_config)

    monkeypatch.setattr(manager, "load_provider", fake_load)

    # Without provider_type, request-time variables pass save-time validation;
    # provider initialization reports misuse for non-chat providers later.
    await manager.create_provider(
        {
            "id": "chat-1",
            "type": "fake_chat",
            "enable": True,
            "custom_headers": {"X-Session": "{sha256:{conversation_id}}"},
        }
    )

    assert config["provider"][0]["id"] == "chat-1"
    assert len(loaded) == 1


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_create_provider_without_provider_type_rejects_bad_syntax(monkeypatch):
    manager, config = _make_provider_manager([])

    async def fail_load(_provider_config):
        pytest.fail("provider must not load when validation fails")

    monkeypatch.setattr(manager, "load_provider", fail_load)

    with pytest.raises(HeaderTemplateError):
        await manager.create_provider(
            {
                "id": "chat-1",
                "type": "fake_chat",
                "enable": True,
                "custom_headers": {"X-Bad": "{unknown}"},
            }
        )

    assert config["provider"] == []
    assert config.saved == 0


@pytest.mark.asyncio
async def test_update_provider_rejects_invalid_header_template(monkeypatch):
    original = {
        "id": "p1",
        "type": "openai_chat_completion",
        "provider_type": "chat_completion",
        "model": "gpt-4o-mini",
        "enable": True,
        "custom_headers": {},
    }
    manager, config = _make_provider_manager([copy.deepcopy(original)])

    async def fail_reload(_provider_config):
        pytest.fail("provider must not reload when validation fails")

    monkeypatch.setattr(manager, "reload", fail_reload)

    with pytest.raises(HeaderTemplateError):
        await manager.update_provider(
            "p1",
            {
                **copy.deepcopy(original),
                "custom_headers": dict(INVALID_TEMPLATE_HEADERS),
            },
        )

    assert config["provider"] == [original]
    assert config.saved == 0


@pytest.mark.asyncio
async def test_update_provider_validates_merged_provider_source_headers(monkeypatch):
    manager, config = _make_provider_manager(
        [
            {
                "id": "p1",
                "provider_source_id": "src-tts",
                "model": "tts-1",
                "enable": True,
            }
        ],
        [
            {
                "id": "src-tts",
                "type": "openai_tts_api",
                "provider_type": "text_to_speech",
                "custom_headers": {"X-Bad": "{umo}"},
            }
        ],
    )

    async def fail_reload(_provider_config):
        pytest.fail("provider must not reload when validation fails")

    monkeypatch.setattr(manager, "reload", fail_reload)

    with pytest.raises(HeaderTemplateError, match="only chat providers support"):
        await manager.update_provider(
            "p1",
            {
                "id": "p1",
                "provider_source_id": "src-tts",
                "model": "tts-1",
                "enable": True,
            },
        )

    assert config["provider"][0]["provider_source_id"] == "src-tts"
    assert config.saved == 0


def _make_config_service(config: dict, provider_manager) -> ProviderConfigService:
    return ProviderConfigService(
        SimpleNamespace(astrbot_config=config, provider_manager=provider_manager)
    )


def _make_dashboard_manager(sources: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        provider_sources_config=sources,
        reload=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_provider_source_upsert_rejects_invalid_template(monkeypatch):
    saved: list[dict] = []
    monkeypatch.setattr(
        config_service_module, "save_config", lambda *args, **kwargs: saved.append({})
    )
    sources = [
        {
            "id": "src1",
            "type": "openai_chat_completion",
            "provider_type": "chat_completion",
            "custom_headers": {"X-Ok": "ok"},
        }
    ]
    config = {
        "provider_sources": copy.deepcopy(sources),
        "provider": [{"id": "p1", "provider_source_id": "src1", "model": "m"}],
    }
    manager = _make_dashboard_manager(config["provider_sources"])
    service = _make_config_service(config, manager)

    with pytest.raises(HeaderTemplateError):
        await service.upsert_provider_source(
            "src1",
            {
                **copy.deepcopy(sources[0]),
                "custom_headers": dict(INVALID_TEMPLATE_HEADERS),
            },
        )

    assert config["provider_sources"] == sources
    assert saved == []
    manager.reload.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_source_upsert_new_source_rejects_invalid_template(monkeypatch):
    saved: list[dict] = []
    monkeypatch.setattr(
        config_service_module, "save_config", lambda *args, **kwargs: saved.append({})
    )
    config = {"provider_sources": [], "provider": []}
    manager = _make_dashboard_manager(config["provider_sources"])
    service = _make_config_service(config, manager)

    with pytest.raises(HeaderTemplateError):
        await service.upsert_provider_source(
            "src-new",
            {
                "id": "src-new",
                "type": "openai_tts_api",
                "provider_type": "text_to_speech",
                "custom_headers": dict(INVALID_TEMPLATE_HEADERS),
            },
        )

    assert config["provider_sources"] == []
    assert saved == []


@pytest.mark.asyncio
async def test_provider_source_upsert_allows_model_value_for_non_chat_source(
    monkeypatch,
):
    saved: list[dict] = []
    monkeypatch.setattr(
        config_service_module, "save_config", lambda *args, **kwargs: saved.append({})
    )
    config = {"provider_sources": [], "provider": []}
    manager = _make_dashboard_manager(config["provider_sources"])
    service = _make_config_service(config, manager)

    await service.upsert_provider_source(
        "src-tts",
        {
            "id": "src-tts",
            "type": "openai_tts_api",
            "provider_type": "text_to_speech",
            "custom_headers": {"X-Model": "{model}"},
        },
    )

    assert config["provider_sources"] == [
        {
            "id": "src-tts",
            "type": "openai_tts_api",
            "provider_type": "text_to_speech",
            "custom_headers": {"X-Model": "{model}"},
        }
    ]
    assert len(saved) == 1


@pytest.mark.asyncio
async def test_provider_source_upsert_validates_model_for_existing_children(
    monkeypatch,
):
    saved: list[dict] = []
    monkeypatch.setattr(
        config_service_module, "save_config", lambda *args, **kwargs: saved.append({})
    )
    sources = [
        {
            "id": "src-tts",
            "type": "openai_tts_api",
            "provider_type": "text_to_speech",
            "custom_headers": {},
        }
    ]
    config = {
        "provider_sources": copy.deepcopy(sources),
        "provider": [{"id": "p1", "provider_source_id": "src-tts", "model": "tts-1"}],
    }
    manager = _make_dashboard_manager(config["provider_sources"])
    service = _make_config_service(config, manager)

    await service.upsert_provider_source(
        "src-tts",
        {**copy.deepcopy(sources[0]), "custom_headers": {"X-Model": "{model}"}},
    )

    assert config["provider_sources"][0]["custom_headers"] == {"X-Model": "{model}"}
    assert len(saved) == 1


@pytest.mark.asyncio
async def test_provider_source_upsert_validates_affected_provider_merge(monkeypatch):
    saved: list[dict] = []
    monkeypatch.setattr(
        config_service_module, "save_config", lambda *args, **kwargs: saved.append({})
    )
    sources = [
        {
            "id": "src1",
            "type": "openai_chat_completion",
            "provider_type": "chat_completion",
            "custom_headers": {},
        }
    ]
    config = {
        "provider_sources": copy.deepcopy(sources),
        "provider": [
            {
                "id": "p1",
                "provider_source_id": "src1",
                "provider_type": "text_to_speech",
                "model": "tts-1",
            }
        ],
    }
    manager = _make_dashboard_manager(config["provider_sources"])
    service = _make_config_service(config, manager)

    with pytest.raises(HeaderTemplateError, match="only chat providers support"):
        await service.upsert_provider_source(
            "src1",
            {**copy.deepcopy(sources[0]), "custom_headers": {"X-Bad": "{umo}"}},
        )

    assert config["provider_sources"] == sources
    assert saved == []
    manager.reload.assert_not_awaited()
