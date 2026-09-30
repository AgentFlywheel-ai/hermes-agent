"""A dead MCP transport session is replaced, never retried.

Each test locks in one rule of the self-heal contract:

- a timed-out tool call asks for a transport rebuild;
- the circuit breaker's half-open probe runs on a fresh session;
- a session-expired retry runs on the new session, not the one that failed;
- the keepalive deadline holds even when the probe ignores cancellation, and
  a reconnect request detaches the old session at once.
"""
import asyncio
import json
import threading
import time
from unittest.mock import MagicMock

import pytest


def _ok_result(text="ok"):
    result = MagicMock()
    result.isError = False
    block = MagicMock()
    block.text = text
    result.content = [block]
    result.structuredContent = None
    return result


def _session(call_tool):
    session = MagicMock()
    session.call_tool = call_tool
    return session


class _StubServer:
    """Minimal server record: a reconnect replaces ``session`` after ``delay``."""

    def __init__(self, name, session, fresh_session=None, delay=0.0):
        self.name = name
        self.session = session
        self._fresh = fresh_session
        self._delay = delay
        self._rpc_lock = asyncio.Lock()
        self._config = {}
        self.reconnects = 0
        self._ready = MagicMock()
        self._ready.is_set.return_value = True
        server = self

        class _Event:
            def set(self_inner):
                server.reconnects += 1
                if server._fresh is None:
                    return

                def _swap():
                    server.session = server._fresh
                    server._rpc_lock = asyncio.Lock()

                if server._delay:
                    threading.Timer(server._delay, _swap).start()
                else:
                    _swap()

        self._reconnect_event = _Event()


@pytest.fixture
def mcp(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool

    mcp_tool._ensure_mcp_loop()
    names = []

    def install(server):
        names.append(server.name)
        mcp_tool._servers[server.name] = server
        mcp_tool._server_error_counts.pop(server.name, None)
        mcp_tool._server_breaker_opened_at.pop(server.name, None)
        return server

    yield mcp_tool, install
    for name in names:
        mcp_tool._servers.pop(name, None)
        mcp_tool._server_error_counts.pop(name, None)
        mcp_tool._server_breaker_opened_at.pop(name, None)


def test_timed_out_tool_call_requests_a_transport_rebuild(mcp):
    mcp_tool, install = mcp

    async def _hangs(*a, **kw):
        await asyncio.sleep(3600)

    server = install(_StubServer("hang", _session(_hangs)))
    handler = mcp_tool._make_tool_handler("hang", "tool", 0.5)

    parsed = json.loads(handler({}))

    assert "timed out" in parsed.get("error", ""), parsed
    # The signal is delivered on the MCP loop thread; give it a moment.
    for _ in range(40):
        if server.reconnects:
            break
        time.sleep(0.05)
    assert server.reconnects == 1


def test_half_open_probe_runs_on_a_fresh_session(mcp, monkeypatch):
    mcp_tool, install = mcp

    async def _dead(*a, **kw):
        raise AssertionError("the probe reused the session that tripped the breaker")

    async def _live(*a, **kw):
        return _ok_result("fresh")

    server = install(_StubServer("probe", _session(_dead), fresh_session=_session(_live)))
    mcp_tool._server_error_counts["probe"] = mcp_tool._CIRCUIT_BREAKER_THRESHOLD
    mcp_tool._server_breaker_opened_at["probe"] = (
        time.monotonic() - mcp_tool._CIRCUIT_BREAKER_COOLDOWN_SEC - 1
    )

    parsed = json.loads(mcp_tool._make_tool_handler("probe", "tool", 10.0)({}))

    assert parsed.get("result") == "fresh", parsed
    assert server.reconnects == 1
    assert mcp_tool._server_error_counts.get("probe", 0) == 0


def test_half_open_probe_without_a_fresh_session_never_touches_the_old_one(mcp, monkeypatch):
    mcp_tool, install = mcp
    monkeypatch.setattr(mcp_tool, "_FRESH_SESSION_WAIT_SEC", 0.3)

    async def _dead(*a, **kw):
        raise AssertionError("the probe reused the session that tripped the breaker")

    server = install(_StubServer("stuck", _session(_dead)))
    mcp_tool._server_error_counts["stuck"] = mcp_tool._CIRCUIT_BREAKER_THRESHOLD
    mcp_tool._server_breaker_opened_at["stuck"] = (
        time.monotonic() - mcp_tool._CIRCUIT_BREAKER_COOLDOWN_SEC - 1
    )
    handler = mcp_tool._make_tool_handler("stuck", "tool", 10.0)

    parsed = json.loads(handler({}))

    assert "reconnect" in parsed.get("error", "").lower(), parsed
    assert server.reconnects == 1
    # The failed probe re-arms the cooldown: the next call short-circuits.
    assert "unreachable" in json.loads(handler({})).get("error", "").lower()


def test_session_expired_retry_waits_for_the_new_session(mcp):
    mcp_tool, install = mcp

    async def _expired(*a, **kw):
        raise RuntimeError("Session terminated")

    async def _live(*a, **kw):
        return _ok_result("new")

    # The rebuild lands after the handler has already started waiting; a
    # retry on the still-attached old session would raise again.
    server = install(_StubServer(
        "exp", _session(_expired), fresh_session=_session(_live), delay=0.5,
    ))

    start = time.monotonic()
    parsed = json.loads(mcp_tool._make_tool_handler("exp", "tool", 10.0)({}))

    assert parsed.get("result") == "new", parsed
    assert server.reconnects == 1
    assert time.monotonic() - start < 5


def test_keepalive_deadline_holds_when_the_probe_ignores_cancellation(mcp, monkeypatch):
    mcp_tool, _ = mcp
    monkeypatch.setattr(mcp_tool, "_MIN_KEEPALIVE_INTERVAL", 0.05)
    monkeypatch.setattr(mcp_tool, "_KEEPALIVE_PROBE_TIMEOUT", 0.2)

    async def _scenario():
        task = mcp_tool.MCPServerTask("ka")
        task._config = {"keepalive_interval": 0.05}

        async def _stubborn_ping():
            # Swallows the first cancellation, the one wait_for relies on;
            # a later cancel (loop shutdown) ends it.
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                pass
            await asyncio.sleep(3600)

        session = MagicMock()
        session.send_ping = _stubborn_ping
        task.session = session
        reason = await asyncio.wait_for(task._wait_for_lifecycle_event(), timeout=5)
        return reason, task.session

    reason, session_after = asyncio.run(_scenario())

    assert reason == "reconnect"
    assert session_after is None


def test_reconnect_request_detaches_the_session_immediately(mcp):
    mcp_tool, _ = mcp

    async def _scenario():
        task = mcp_tool.MCPServerTask("detach")
        task._config = {}
        task.session = MagicMock()
        task._reconnect_event.set()
        reason = await task._wait_for_lifecycle_event()
        return reason, task.session

    reason, session_after = asyncio.run(_scenario())

    assert reason == "reconnect"
    assert session_after is None
