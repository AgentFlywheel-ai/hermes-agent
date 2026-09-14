"""per_turn_credential: the bearer rides tools/call only, and a call without one is refused."""
import json
from unittest.mock import MagicMock

from gateway.session_context import clear_session_vars, set_session_vars
from tools import mcp_tool
from tools.mcp_tool import _make_tool_handler, _per_call_bearer_auth


def _fake_server(per_turn: bool):
    server = MagicMock()
    server.name = "srv"
    server._config = {"per_turn_credential": per_turn}
    server._call_bearer = None
    server._ready = MagicMock()
    server._ready.is_set.return_value = True
    seen = {}

    async def call_tool(name, arguments=None):
        seen["bearer_during_call"] = server._call_bearer
        result = MagicMock()
        result.isError = False
        result.content = []
        result.structuredContent = None
        return result

    server.session = MagicMock()
    server.session.call_tool = call_tool
    return server, seen


def _install(server):
    mcp_tool._servers["srv"] = server
    mcp_tool._server_error_counts.pop("srv", None)
    mcp_tool._ensure_mcp_loop()


def test_opt_in_server_refuses_a_call_with_no_credential_bound():
    server, seen = _fake_server(True)
    _install(server)
    try:
        out = json.loads(_make_tool_handler("srv", "lookup", 10.0)({"q": 1}))
        assert out.get("unavailable") is True
        assert "credential" in out["error"]
        assert "bearer_during_call" not in seen
    finally:
        mcp_tool._servers.pop("srv", None)


def test_opt_in_server_carries_the_bearer_only_during_the_call():
    server, seen = _fake_server(True)
    _install(server)
    tokens = set_session_vars(platform="api_server", tool_credential="tok-9")
    try:
        _make_tool_handler("srv", "lookup", 10.0)({"q": 1})
        assert seen["bearer_during_call"] == "tok-9"
        assert server._call_bearer is None
    finally:
        clear_session_vars(tokens)
        mcp_tool._servers.pop("srv", None)


def test_server_without_opt_in_never_sees_the_credential():
    server, seen = _fake_server(False)
    _install(server)
    tokens = set_session_vars(platform="api_server", tool_credential="tok-9")
    try:
        _make_tool_handler("srv", "lookup", 10.0)({"q": 1})
        assert seen["bearer_during_call"] is None
    finally:
        clear_session_vars(tokens)
        mcp_tool._servers.pop("srv", None)


def test_auth_flow_overrides_authorization_only_while_a_bearer_is_set():
    import httpx

    server = MagicMock()
    server._call_bearer = None
    auth = _per_call_bearer_auth(server)
    request = httpx.Request("POST", "https://product.example/mcp", headers={"Authorization": "Bearer discovery"})
    assert next(auth.auth_flow(request)).headers["Authorization"] == "Bearer discovery"
    server._call_bearer = "tok"
    request = httpx.Request("POST", "https://product.example/mcp", headers={"Authorization": "Bearer discovery"})
    assert next(auth.auth_flow(request)).headers["Authorization"] == "Bearer tok"
