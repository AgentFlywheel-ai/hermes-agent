"""An MCP server that does not answer at gateway startup is retried.

A server that accepts the TCP connection but never answers (for example while
it is being redeployed behind a proxy) exhausts ``connect_timeout`` during
startup discovery. That path never reached the server task's own
initial-connect retry, so the server stayed unregistered until the process
was restarted and no give-up line was ever logged.

The recovery test runs a real FastMCP server in a subprocess; the first
connect is made against a socket that accepts and never answers.
"""
import json
import logging
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
    mcp = FastMCP("startup-probe")

    @mcp.tool()
    def echo(text: str) -> str:
        return "echo:" + text

    uvicorn.run(mcp.streamable_http_app(), host="127.0.0.1",
                port=int(sys.argv[1]), log_level="error")
    """
)

NAME = "startupprobe"
GIVE_UP = f"MCP server '{NAME}' failed initial connection after"


def _silent_listener() -> socket.socket:
    """A socket that accepts connections and never answers them."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    return sock


def _wait_for(predicate, deadline: float) -> bool:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _config(port: int, connect_timeout: float) -> dict:
    return {NAME: {"url": f"http://127.0.0.1:{port}/mcp", "connect_timeout": connect_timeout}}


@pytest.fixture
def mcp_tool(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool as module

    try:
        yield module
    finally:
        module.shutdown_mcp_servers()
        module._server_connect_errors.pop(NAME, None)


def test_server_silent_at_startup_registers_its_tools_once_it_answers(mcp_tool):
    silent = _silent_listener()
    port = silent.getsockname()[1]
    proc = None
    try:
        mcp_tool.register_mcp_servers(_config(port, connect_timeout=1))
        assert NAME not in mcp_tool._servers
        assert NAME in mcp_tool._startup_retry_tasks

        silent.close()
        proc = subprocess.Popen([sys.executable, "-c", _SERVER_SRC, str(port)])

        assert _wait_for(lambda: NAME in mcp_tool._servers, 30), "never recovered"
        registered = mcp_tool._servers[NAME]._registered_tool_names
        assert f"mcp_{NAME}_echo" in registered
        assert f"mcp_{NAME}_echo" in mcp_tool._existing_tool_names()
        assert _wait_for(lambda: NAME not in mcp_tool._startup_retry_tasks, 5)
        assert NAME not in mcp_tool._server_connect_errors

        handler = mcp_tool._make_tool_handler(NAME, "echo", 20.0)
        assert json.loads(handler({"text": "hi"})).get("result") == "echo:hi"
    finally:
        silent.close()
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=10)


def test_server_refusing_the_first_connect_registers_its_tools_once_it_accepts(mcp_tool, caplog):
    """Nothing listens when discovery starts; the server comes up shortly
    after and is picked up by the server task's initial-connect retries."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    late_start = "import time; time.sleep(0.5)\n" + _SERVER_SRC
    proc = subprocess.Popen([sys.executable, "-c", late_start, str(port)])
    try:
        with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
            mcp_tool.register_mcp_servers(_config(port, connect_timeout=30))
        messages = [r.getMessage() for r in caplog.records]
        assert any("initial connection failed (attempt 1/" in m for m in messages), messages
        assert not any("giving up" in m for m in messages), messages
        assert f"mcp_{NAME}_echo" in mcp_tool._servers[NAME]._registered_tool_names
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_server_that_never_answers_gives_up_once(mcp_tool, monkeypatch, caplog):
    monkeypatch.setattr(mcp_tool, "_MAX_BACKOFF_SECONDS", 0.05)
    silent = _silent_listener()
    try:
        with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
            mcp_tool.register_mcp_servers(_config(silent.getsockname()[1], connect_timeout=0.3))
            assert NAME in mcp_tool._startup_retry_tasks
            assert _wait_for(lambda: NAME not in mcp_tool._startup_retry_tasks, 20)
    finally:
        silent.close()

    give_ups = [r.getMessage() for r in caplog.records if "giving up" in r.getMessage()]
    assert len(give_ups) == 1, give_ups
    assert give_ups[0].startswith(GIVE_UP)
    assert NAME not in mcp_tool._servers
    assert NAME not in mcp_tool._server_connecting
    assert NAME in mcp_tool._server_connect_errors


def test_failure_reported_by_the_server_task_is_not_retried_again(mcp_tool, monkeypatch):
    """The server task runs its own initial-connect retries and logs its own
    give-up line; a second layer of retries would log it twice."""
    calls = []

    async def _refused(name, config):
        calls.append(name)
        raise ConnectionError("refused")

    monkeypatch.setattr(mcp_tool, "_connect_server", _refused)
    mcp_tool.register_mcp_servers(_config(1, connect_timeout=5))

    assert calls == [NAME]
    assert NAME not in mcp_tool._startup_retry_tasks
    assert mcp_tool._server_connect_errors[NAME] == "refused"


def test_startup_timeout_does_not_leave_the_server_task_running(mcp_tool):
    """Each timed-out attempt cancels its own server task instead of leaving
    a hung transport behind for every retry."""
    import asyncio

    silent = _silent_listener()
    try:
        mcp_tool.register_mcp_servers(_config(silent.getsockname()[1], connect_timeout=0.3))

        async def _server_tasks():
            return [
                t for t in asyncio.all_tasks()
                if getattr(t.get_coro(), "__qualname__", "") == "MCPServerTask.run"
            ]

        assert mcp_tool._run_on_mcp_loop(_server_tasks, timeout=5) == []
    finally:
        silent.close()


def test_shutdown_cancels_a_pending_startup_retry(mcp_tool):
    silent = _silent_listener()
    try:
        mcp_tool.register_mcp_servers(_config(silent.getsockname()[1], connect_timeout=0.3))
        task = mcp_tool._startup_retry_tasks[NAME]

        mcp_tool.shutdown_mcp_servers()

        assert task.done()
        assert mcp_tool._startup_retry_tasks == {}
        assert NAME not in mcp_tool._server_connecting
        assert mcp_tool._mcp_loop is None
    finally:
        silent.close()
