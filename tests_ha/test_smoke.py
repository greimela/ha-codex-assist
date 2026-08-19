"""Smoke tests against a real Home Assistant instance.

These verify the integration wires into real HA APIs (config entries,
conversation platform, AI Task platform, chat log streaming) instead of the
lightweight fakes used by the main test suite. Only the Codex backend HTTP
calls are stubbed.
"""

from __future__ import annotations

import pytest
from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import DATA_CHAT_LOGS
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import Context, HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.codex_assist import DOMAIN
from custom_components.codex_assist.codex_client import CodexClient, CodexTextDelta
from custom_components.codex_assist.conversation import (
    _stream_codex_turn_into_chat_log,
)
from custom_components.codex_assist.diagnostics import (
    REDACTED,
    async_get_config_entry_diagnostics,
)

ENTRY_DATA = {
    # Not a JWT, so the runtime treats it as non-expiring and skips refresh.
    "access_token": "test-access-token",
    "refresh_token": "test-refresh-token",
    "model": "gpt-5.4",
    "prompt": "You are a concise Home Assistant Assist conversation agent.",
    "web_search": True,
}


def _make_entry() -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN,
        title="Codex Assist",
        unique_id=DOMAIN,
        data=dict(ENTRY_DATA),
    )


async def _setup_entry(hass: HomeAssistant) -> MockConfigEntry:
    # The conversation component requires the core homeassistant component
    # (exposed-entities registry) to be set up first.
    assert await async_setup_component(hass, "homeassistant", {})
    entry = _make_entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def test_setup_creates_conversation_and_ai_task_entities(
    hass: HomeAssistant,
) -> None:
    entry = await _setup_entry(hass)

    assert entry.state is ConfigEntryState.LOADED
    assert hass.states.get("conversation.codex_assist") is not None
    assert hass.states.get("ai_task.codex_assist_ai_task") is not None


async def test_unload_entry_cleans_up(hass: HomeAssistant) -> None:
    entry = await _setup_entry(hass)

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.NOT_LOADED


async def test_conversation_turn_streams_codex_reply(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _setup_entry(hass)

    async def fake_stream_turn(self: CodexClient, **kwargs: object):
        yield CodexTextDelta("The porch light is on.")

    monkeypatch.setattr(CodexClient, "stream_turn", fake_stream_turn)

    result = await conversation.async_converse(
        hass,
        "Is the porch light on?",
        None,
        Context(),
        agent_id="conversation.codex_assist",
    )

    speech = result.response.speech["plain"]["speech"]
    assert speech == "The porch light is on."


async def test_conversation_turn_omits_trailing_sources_from_speech(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _setup_entry(hass)

    cited_response = (
        "Die Kramerei am Kreisel in Dorfen hat heute, Mittwoch, den "
        "19. August 2026, von 6 Uhr bis 18 Uhr geöffnet. "
        "([kramerei-am-kreisel.de](https://www.kramerei-am-kreisel.de/"
        "kontakt?utm_source=openai))"
    )

    async def fake_stream_turn(self: CodexClient, **kwargs: object):
        yield CodexTextDelta(cited_response)

    monkeypatch.setattr(CodexClient, "stream_turn", fake_stream_turn)

    result = await conversation.async_converse(
        hass,
        "What will the weather be tomorrow?",
        None,
        Context(),
        agent_id="conversation.codex_assist",
    )

    speech = result.response.speech["plain"]["speech"]
    assert speech == (
        "Die Kramerei am Kreisel in Dorfen hat heute, Mittwoch, den "
        "19. August 2026, von 6 Uhr bis 18 Uhr geöffnet."
    )
    assert result.conversation_id is not None
    assert hass.data[DATA_CHAT_LOGS][result.conversation_id].content[-1].content == (
        "Die Kramerei am Kreisel in Dorfen hat heute, Mittwoch, den "
        "19. August 2026, von 6 Uhr bis 18 Uhr geöffnet."
    )


async def test_web_search_citations_do_not_reach_streaming_tts_input(
    hass: HomeAssistant,
) -> None:
    streamed_deltas: list[dict] = []
    chat_log = conversation.ChatLog(
        hass,
        "streaming-tts-test",
        delta_listener=lambda _chat_log, delta: streamed_deltas.append(delta),
    )
    cited_response = (
        "Die Kramerei am Kreisel in Dorfen hat heute von 6 Uhr bis 18 Uhr geöffnet. "
        "([kramerei-am-kreisel.de](https://www.kramerei-am-kreisel.de/"
        "kontakt?utm_source=openai))"
    )

    class FakeCodex:
        async def stream_turn(self, **kwargs: object):
            yield CodexTextDelta(cited_response[:90])
            yield CodexTextDelta(cited_response[90:])

    await _stream_codex_turn_into_chat_log(
        chat_log=chat_log,
        codex=FakeCodex(),
        entity_id="conversation.codex_assist",
        model="gpt-5.4",
        instructions="Be concise.",
        input_items=[{"role": "user", "content": "When is the shop open?"}],
        tools=[{"type": "web_search"}],
        reasoning_effort="low",
        reasoning_summary="auto",
        text_verbosity="medium",
    )

    tts_input = "".join(
        delta.get("content", "")
        for delta in streamed_deltas
        if delta.get("role") in (None, "assistant")
    )
    assert tts_input == (
        "Die Kramerei am Kreisel in Dorfen hat heute von 6 Uhr bis 18 Uhr geöffnet."
    )


async def test_diagnostics_redact_tokens_on_real_entry(hass: HomeAssistant) -> None:
    entry = await _setup_entry(hass)

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    entry_data = diagnostics["entry"]["data"]
    assert entry_data["access_token"] == REDACTED
    assert entry_data["refresh_token"] == REDACTED
    assert entry_data["model"] == "gpt-5.4"
    assert "test-access-token" not in str(diagnostics)
