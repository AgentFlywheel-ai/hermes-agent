"""A long-running client survives a Streamable-HTTP MCP server restart.

These tests run a real FastMCP server in a subprocess and restart it under a
connected client. They cover the path unit stubs cannot: a server restart
invalidates every session id it issued, and the client must replace its
transport session instead of reusing the dead one.
"""
import json
import socket
import subprocess
import sys
import textwrap
import time

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("uvicorn")

_SERVER_SRC = textwrap.dedent(
    """
    import sys, uvicorn
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("restart-probe")

    @mcp.tool()
    def echo(text: str) -> str:
        return "echo:" + text

    uvicorn.run(mcp.streamable_http_app(), host="127.0.0.1",
                port=int(sys.argv[1]), log_level="error")
    """
)

TOOL_TIMEOUT = 20.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_listening(port: int, deadline: float = 15.0) -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"server on {port} never listened")


class _Server:
    def __init__(self, port: int):
        self.port = port
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen([sys.executable, "-c", _SERVER_SRC, str(self.port)])
        _wait_listening(self.port)

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait(timeout=10)
            self.proc = None


@pytest.fixture
def live_server(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool

    server = _Server(_free_port())
    server.start()
    name = "restartprobe"
    mcp_tool._server_error_counts.pop(name, None)
    mcp_tool._server_breaker_opened_at.pop(name, None)
    try:
        mcp_tool.register_mcp_servers({
            name: {
                "url": f"http://127.0.0.1:{server.port}/mcp",
                "timeout": TOOL_TIMEOUT,
            }
        })
        assert name in mcp_tool._servers, "client never connected"
        yield server, name
    finally:
        mcp_tool.shutdown_mcp_servers()
        mcp_tool._server_error_counts.pop(name, None)
        mcp_tool._server_breaker_opened_at.pop(name, None)
        server.stop()


def _timed_call(handler, text):
    start = time.monotonic()
    out = json.loads(handler({"text": text}))
    return out, time.monotonic() - start


def test_first_call_after_server_restart_succeeds_on_a_new_session(live_server):
    """The call that discovers the restart reconnects and retries on the
    NEW session. It must not retry on the dead session and wait out the
    tool timeout."""
    from tools import mcp_tool

    server, name = live_server
    handler = mcp_tool._make_tool_handler(name, "echo", TOOL_TIMEOUT)
    first, _ = _timed_call(handler, "before")
    assert first.get("result") == "echo:before"
    old_session = mcp_tool._servers[name].session

    server.stop()
    server.start()

    out, elapsed = _timed_call(handler, "after")
    assert out.get("result") == "echo:after", out
    assert elapsed < TOOL_TIMEOUT / 2, f"recovery call took {elapsed:.1f}s"
    assert mcp_tool._servers[name].session is not old_session


def test_a_wedged_session_is_replaced_after_one_timed_out_call(live_server):
    """Whatever wedges a live session (here: an RPC lock that is never
    released), one timed-out call rebuilds the transport and the next call
    runs on a new session."""
    import asyncio
    from tools import mcp_tool

    _, name = live_server
    server = mcp_tool._servers[name]
    old_session = server.session

    async def _hold_forever():
        await server._rpc_lock.acquire()
        await asyncio.Event().wait()

    asyncio.run_coroutine_threadsafe(_hold_forever(), mcp_tool._mcp_loop)

    handler = mcp_tool._make_tool_handler(name, "echo", 2.0)
    wedged = json.loads(handler({"text": "wedged"}))
    assert "timed out" in wedged.get("error", ""), wedged

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        live = server.session
        if live is not None and live is not old_session:
            break
        time.sleep(0.1)
    out, elapsed = _timed_call(handler, "healed")
    assert out.get("result") == "echo:healed", out
    assert server.session is not old_session


def test_calls_after_restart_never_wait_out_the_tool_timeout(live_server):
    """Every call after a restart is fast: the recovering call and the
    ones after it."""
    from tools import mcp_tool

    server, name = live_server
    handler = mcp_tool._make_tool_handler(name, "echo", TOOL_TIMEOUT)
    assert _timed_call(handler, "warm")[0].get("result") == "echo:warm"

    server.stop()
    server.start()

    for i in range(3):
        out, elapsed = _timed_call(handler, f"n{i}")
        assert out.get("result") == f"echo:n{i}", out
        assert elapsed < TOOL_TIMEOUT / 2, f"call {i} took {elapsed:.1f}s"
