"""``tools.exclude_match: casefold`` withholds every spelling of an excluded tool.

A server that publishes its own catalog is bound without ``tools.include``,
so its exclusions are compared against names the operator never saw. With the
opt-in, exclusion equality is ``str.strip().casefold()`` on both sides, for
initial discovery and for ``tools/list_changed`` refreshes alike. Without it,
matching stays literal.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tools.mcp_tool import MCPServerTask, _register_server_tools
from tools.registry import ToolRegistry


def _tool(name):
    return SimpleNamespace(name=name, description="", inputSchema=None)


def _server(name, tool_names):
    server = MCPServerTask(name)
    server._tools = [_tool(n) for n in tool_names]
    # A capability-free session: only the server's own tools register.
    server.session = SimpleNamespace()
    return server


def _register(name, tool_names, tools_config):
    registry = ToolRegistry()
    server = _server(name, tool_names)
    with patch("tools.registry.registry", registry):
        registered = _register_server_tools(name, server, {"tools": tools_config})
    return registered


def test_casefold_exclude_withholds_every_spelling():
    registered = _register(
        "cat",
        ["lookup_record", "Lookup_Record", "LOOKUP_RECORD", "search_records"],
        {"exclude": ["  Lookup_RECORD "], "exclude_match": "casefold"},
    )
    assert registered == ["mcp_cat_search_records"]


def test_casefold_exclude_accepts_normalized_duplicates():
    registered = _register(
        "dup",
        ["lookup_record", "search_records"],
        {"exclude": ["lookup_record", "LOOKUP_RECORD", " lookup_record"],
         "exclude_match": "casefold"},
    )
    assert registered == ["mcp_dup_search_records"]


def test_casefold_keeps_the_discovered_spelling_for_calls():
    registry = ToolRegistry()
    server = _server("keep", ["Search_Records", "lookup_record"])
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._make_tool_handler") as make_handler:
        _register_server_tools(
            "keep", server,
            {"tools": {"exclude": ["LOOKUP_RECORD"], "exclude_match": "casefold"}},
        )
    called_with = [call.args[1] for call in make_handler.call_args_list]
    assert called_with == ["Search_Records"]


def test_literal_exclude_is_unchanged_without_the_opt_in():
    registered = _register(
        "lit",
        ["lookup_record", "search_records"],
        {"exclude": ["LOOKUP_RECORD"]},
    )
    assert registered == ["mcp_lit_lookup_record", "mcp_lit_search_records"]


def test_include_matching_is_unchanged_by_the_opt_in():
    registered = _register(
        "inc",
        ["search_records", "Search_Records"],
        {"include": ["search_records"], "exclude_match": "casefold"},
    )
    assert registered == ["mcp_inc_search_records"]


def test_unknown_exclude_match_value_keeps_literal_matching():
    registered = _register(
        "bad",
        ["lookup_record", "search_records"],
        {"exclude": ["LOOKUP_RECORD"], "exclude_match": "fuzzy"},
    )
    assert registered == ["mcp_bad_lookup_record", "mcp_bad_search_records"]


@pytest.mark.asyncio
async def test_refresh_applies_casefold_exclude():
    registry = ToolRegistry()
    server = _server("live", ["search_records"])
    server._config = {"tools": {"exclude": ["lookup_record"], "exclude_match": "casefold"}}
    server._refresh_lock = asyncio.Lock()
    with patch("tools.registry.registry", registry):
        server._registered_tool_names = _register_server_tools("live", server, server._config)
        server.session = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(
            tools=[_tool("search_records"), _tool("LOOKUP_Record")],
        )))
        with patch.object(MCPServerTask, "_advertises_tools", return_value=True):
            await server._refresh_tools()
        assert server._registered_tool_names == ["mcp_live_search_records"]
