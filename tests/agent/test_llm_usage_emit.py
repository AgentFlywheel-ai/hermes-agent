"""Tests for agent/llm_usage_emit.py — m2-llm-spend-ledger Seam 2 (U3 / contract C2).

Covers the fork's only m2 change: emitting one tagged ``LLM_USAGE_EVENT`` line per completed
Bedrock turn. Asserts C1 conformance (the captured-line fixture assertion U3 owns), the
triage-identical correlation id, the raw-stdout tagged format, and the fail-open guarantee.
"""

import json
from types import SimpleNamespace

import pytest

from agent.llm_usage_emit import (
    TAG,
    build_event,
    compute_correlation_id,
    emit_bedrock_turn,
    format_event_line,
    slug,
)

# The C1 "required" field set (event is rejected by the validator without these). Mirrors
# agentflywheel compose/shared/llm_usage.schema.json#required — vendored here because the fork
# cannot import the afai-side schema. The agentflywheel-side PR keeps these in lockstep.
C1_REQUIRED = {
    "venture",
    "source",
    "archetype",
    "role_or_service",
    "correlation_id",
    "unit_type",
    "created_at",
    "harness",
    "llm_transport",
    "model_provider",
    "model",
    "tokens_in",
    "tokens_out",
    "schema_version",
}


def _zulip_stream_agent():
    """A minimal stand-in for an AIAgent mid-Bedrock-turn on a Zulip stream message."""
    return SimpleNamespace(
        platform="zulip",
        model="us.anthropic.claude-opus-4-8",
        _chat_type="stream",
        _chat_name="bugs",
        _chat_id="42:Login flow is BROKEN",
        _thread_id="Login flow is BROKEN",
    )


def _bedrock_response():
    return SimpleNamespace(
        model="us.anthropic.claude-opus-4-8",
        usage=SimpleNamespace(prompt_tokens=1234, completion_tokens=567, total_tokens=1801),
    )


# --- slug / correlation_id converge with triage compute_external_id ---------------------


def test_slug_matches_triage_normalization():
    # Same cases the triage slug() guarantees: casing/whitespace/Unicode collapse.
    assert slug("Login flow is BROKEN") == "login-flow-is-broken"
    assert slug("  Spaced   Out  ") == "spaced-out"
    assert slug("Hello, World!") == "hello-world"
    assert slug("") == ""


def test_correlation_id_form_matches_triage():
    # zulip:<realm>:<stream>:<topic_slug> — identical to compute_external_id(realm, stream, topic).
    assert (
        compute_correlation_id("movementlens", "bugs", "Login flow is BROKEN")
        == "zulip:movementlens:bugs:login-flow-is-broken"
    )


# --- build_event: C1 conformance + hermes constants -------------------------------------


def test_build_event_carries_c1_required_set_and_hermes_constants():
    ev = build_event(
        realm="movementlens",
        platform="zulip",
        chat_type="stream",
        stream="bugs",
        topic="Login flow is BROKEN",
        chat_id="42:Login flow is BROKEN",
        model="us.anthropic.claude-opus-4-8",
        tokens_in=1234,
        tokens_out=567,
    )
    assert C1_REQUIRED <= set(ev)  # every required field present
    assert ev["archetype"] == "hermes-persistent"
    assert ev["harness"] == "hermes-runtime"
    assert ev["llm_transport"] == "bedrock"
    assert ev["source"] == "afai-agent"
    assert ev["role_or_service"] == "agent-a"
    assert ev["unit_type"] == "turn"
    assert ev["model_provider"] == "anthropic"
    assert ev["venture"] == "movementlens"
    assert ev["correlation_id"] == "zulip:movementlens:bugs:login-flow-is-broken"
    assert ev["tokens_in"] == 1234 and ev["tokens_out"] == 567
    assert ev["cache_tokens"] == 0
    assert ev["schema_version"] == "1"


def test_non_stream_turn_falls_back_to_platform_chat_id():
    ev = build_event(
        realm="movementlens",
        platform="zulip",
        chat_type="dm",
        stream="someone@example.com",
        topic=None,
        chat_id="dm:someone@example.com",
        model="us.anthropic.claude-opus-4-8",
        tokens_in=1,
        tokens_out=2,
    )
    assert ev["correlation_id"] == "zulip:dm:someone@example.com"


# --- format_event_line: tagged, compact, round-trips ------------------------------------


def test_format_event_line_is_tagged_and_parses():
    ev = build_event(
        realm="movementlens",
        platform="zulip",
        chat_type="stream",
        stream="bugs",
        topic="x",
        chat_id="1:x",
        model="m",
        tokens_in=1,
        tokens_out=2,
    )
    line = format_event_line(ev)
    # The Vector route matches starts_with(.message, "LLM_USAGE_EVENT ").
    assert line.startswith(TAG + " ")
    payload = line[len(TAG) + 1:]
    assert json.loads(payload) == ev  # compact JSON round-trips


# --- emit_bedrock_turn: the captured-line fixture assertion (C1) -------------------------


def test_emit_writes_one_c1_valid_line(capsys, monkeypatch):
    monkeypatch.setenv("TENANT_NAME", "movementlens")
    emit_bedrock_turn(_zulip_stream_agent(), _bedrock_response())

    out = capsys.readouterr().err.strip()  # stderr: the channel that reaches the docker stream under s6
    assert out.startswith(TAG + " ")  # exactly the tagged line Vector tails

    event = json.loads(out[len(TAG) + 1:])
    # Fixture assertion: the captured emitted line carries the full C1 required set ...
    assert C1_REQUIRED <= set(event)
    # ... with the hermes-persistent identity and the triage-converged correlation id.
    assert event["archetype"] == "hermes-persistent"
    assert event["harness"] == "hermes-runtime"
    assert event["llm_transport"] == "bedrock"
    assert event["correlation_id"] == "zulip:movementlens:bugs:login-flow-is-broken"
    assert event["tokens_in"] == 1234 and event["tokens_out"] == 567


def test_emit_is_fail_open_on_broken_inputs(capsys):
    # A response with no usage / a bare object must never raise and must not emit a usage line.
    emit_bedrock_turn(object(), object())
    captured = capsys.readouterr()
    assert captured.out == ""
    assert TAG not in captured.err  # only a non-tagged AFAI_USAGE_DEBUG line, never a usage row


def test_emit_never_raises_even_if_stderr_explodes(monkeypatch):
    # Even if the write itself fails, the turn must not see an exception.
    import agent.llm_usage_emit as m

    monkeypatch.setattr(m.sys.stderr, "write", lambda *_: (_ for _ in ()).throw(IOError("boom")))
    emit_bedrock_turn(_zulip_stream_agent(), _bedrock_response())  # no raise == pass
