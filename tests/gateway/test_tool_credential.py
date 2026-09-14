"""Per-turn tool credential: bound per API-server turn, never exported, never echoed."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.session_context import _VAR_MAP, clear_session_vars, get_tool_credential, set_session_vars
from tests.gateway.test_api_server import _create_app, _make_adapter


def test_credential_is_bound_cleared_and_outside_the_env_bridge_map():
    assert get_tool_credential() == ""
    tokens = set_session_vars(platform="api_server", tool_credential="tok-1")
    try:
        assert get_tool_credential() == "tok-1"
    finally:
        clear_session_vars(tokens)
    assert get_tool_credential() == ""
    # The subprocess env bridge iterates _VAR_MAP; the credential must never be in it.
    assert "HERMES_SESSION_TOOL_CREDENTIAL" not in _VAR_MAP


@pytest.mark.asyncio
async def test_credential_reaches_run_agent_and_is_not_echoed():
    adapter = _make_adapter(api_key="sk-secret")
    mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            resp = await cli.post(
                "/v1/chat/completions",
                headers={"X-Hermes-Tool-Credential": "tok-abc", "Authorization": "Bearer sk-secret"},
                json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status == 200
        assert mock_run.call_args.kwargs["tool_credential"] == "tok-abc"
        assert "X-Hermes-Tool-Credential" not in resp.headers
        assert "tok-abc" not in await resp.text()


@pytest.mark.asyncio
async def test_credential_absent_yields_none():
    adapter = _make_adapter(api_key="sk-secret")
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ({"final_response": "ok", "messages": [], "api_calls": 1},
                                     {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            resp = await cli.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-secret"},
                json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status == 200
        assert mock_run.call_args.kwargs["tool_credential"] is None


@pytest.mark.asyncio
async def test_credential_rejected_without_api_key():
    adapter = _make_adapter()
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/chat/completions",
            headers={"X-Hermes-Tool-Credential": "whatever"},
            json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status == 403


def test_credential_rejects_control_chars_and_oversize():
    adapter = _make_adapter(api_key="sk-secret")
    request = MagicMock()
    request.headers = {"X-Hermes-Tool-Credential": "bad\rvalue"}
    cred, err = adapter._parse_tool_credential_header(request)
    assert cred is None and err is not None and err.status == 400
    request.headers = {"X-Hermes-Tool-Credential": "x" * 5000}
    cred, err = adapter._parse_tool_credential_header(request)
    assert cred is None and err is not None and err.status == 400
    request.headers = {"X-Hermes-Tool-Credential": "eyJ.ok-token_value"}
    cred, err = adapter._parse_tool_credential_header(request)
    assert cred == "eyJ.ok-token_value" and err is None
