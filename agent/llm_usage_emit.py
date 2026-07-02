"""
llm_usage_emit.py — m2-llm-spend-ledger Seam 2 (U3 / contract C2): emit one tagged
``llm_usage`` line per completed Bedrock turn for the hermes-persistent archetype (Agent A).

This is the afai-owned fork's *only* m2 change. Per the C1/C2 contracts (in the agentflywheel
repo: ``.afai/programs/agent-platform/m2-llm-spend-ledger/CONTRACTS.md``):

- **stdout only.** Each completed Bedrock turn writes one ``LLM_USAGE_EVENT {compact-json}``
  line to the process stdout. The tenant VM's existing Vector recognizes the tag, parses the
  JSON, and ships it to the central OpenObserve ``llm_usage`` stream. This module holds **no**
  credential, opens **no** socket, and never POSTs — Vector owns buffering + delivery.
- **fail-open by construction.** :func:`emit_bedrock_turn` wraps everything in a bare
  ``try/except`` so a malformed turn degrades to a dropped line and can **never** raise into,
  block, or slow the agent's hot path. The turn behaves exactly as before this module existed.
- **raw line, not via ``logging``.** The Vector route matches ``starts_with(.message,
  "LLM_USAGE_EVENT ")``; routing the line through Python ``logging`` would prepend a
  timestamp/level and break that prefix match. So we ``print`` the bare tagged line.

``correlation_id`` convergence: for a Zulip stream turn the id is
``zulip:<realm>:<channel>:<topic_slug>``, byte-identical to the triage supervisor's
``compute_external_id()`` / ``slug()`` (agentflywheel ``compose/tenant/triage/
triage_supervisor.py``) so a topic's triage task and the Hermes turns in that topic share one
correlation lineage. ``realm`` is the tenant realm string_id — the ``TENANT_NAME`` env value,
the same source triage uses (wired into the hermes container env in ``hermes-stack.yml``).

stdlib only — no new dependency reaches the pinned image.
"""

import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone

#: Tag the tenant Vector's ``llm_usage_route`` matches at the start of the stdout line.
TAG = "LLM_USAGE_EVENT"

#: C1 event schema version (mirrors agentflywheel compose/shared/llm_usage.schema.json).
SCHEMA_VERSION = "1"


def slug(topic):
    """Normalize a Zulip topic into a URL-safe slug.

    Byte-for-byte the same normalization as triage's ``slug()`` so the two seams compute an
    identical ``external_id`` for the same topic: NFKC → casefold → strip → whitespace runs to
    ``-`` → drop chars outside ``[a-z0-9-]`` → collapse repeated ``-`` → strip leading/trailing
    ``-``.
    """
    if not topic:
        return ""
    s = unicodedata.normalize("NFKC", topic)
    s = s.casefold()
    s = s.strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9-]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    s = s.strip("-")
    return s


def compute_correlation_id(realm, stream, topic):
    """``zulip:<realm>:<stream>:<topic_slug>`` — identical form to triage compute_external_id()."""
    return f"zulip:{realm}:{stream}:{slug(topic)}"


def _model_provider(model):
    """Best-effort C1 ``model_provider`` enum for a Bedrock model id.

    Agent A runs Claude on Bedrock → ``anthropic``. Anything we can't confidently classify
    falls to the C1 ``unknown`` sentinel (a member of the enum) rather than a bogus value, so
    the C4 rate-table join is never poisoned.
    """
    m = (model or "").lower()
    if "anthropic" in m or "claude" in m:
        return "anthropic"
    if "meta" in m or "llama" in m:
        return "meta"
    if "openai" in m or "gpt" in m:
        return "openai"
    if "google" in m or "gemini" in m:
        return "google"
    return "unknown"


def build_event(
    *,
    realm,
    platform,
    chat_type,
    stream,
    topic,
    chat_id,
    model,
    tokens_in,
    tokens_out,
    cache_tokens=0,
):
    """Build one C1-shaped ``llm_usage`` event dict for a completed Bedrock turn.

    Pure — no I/O, no env reads. The hermes-persistent constants are fixed here; the per-turn
    fields come from the caller. For a Zulip stream turn ``correlation_id`` is the
    ``(realm, channel, topic)`` triple; otherwise it degrades to ``<platform>:<chat_id>`` so the
    id is never empty (a DM/cron turn still rolls up, just not into a triage topic lineage).
    """
    if platform == "zulip" and chat_type == "stream" and stream and topic:
        correlation_id = compute_correlation_id(realm, stream, topic)
    else:
        correlation_id = f"{platform or 'unknown'}:{chat_id or 'unknown'}"

    return {
        # identity / attribution
        "venture": realm or "unknown",
        "source": "afai-agent",
        "archetype": "hermes-persistent",
        "role_or_service": "agent-a",
        # correlation
        "correlation_id": correlation_id,
        "unit_type": "turn",
        "trigger": "message",
        # timing
        "created_at": datetime.now(timezone.utc).isoformat(),
        # the five model-call axes
        "harness": "hermes-runtime",
        "llm_transport": "bedrock",
        "model_provider": _model_provider(model),
        "model": str(model or "unknown"),
        # usage
        "tokens_in": int(tokens_in or 0),
        "tokens_out": int(tokens_out or 0),
        "cache_tokens": int(cache_tokens or 0),
        # version
        "schema_version": SCHEMA_VERSION,
    }


def format_event_line(event):
    """Serialize ``event`` to the tagged line ``LLM_USAGE_EVENT {compact-json}``.

    Compact separators + ``sort_keys=True`` so the payload round-trips through the agentflywheel
    ``llm_usage.parse_event_line()`` helper exactly. Pure and never raises.
    """
    payload = json.dumps(event, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    return f"{TAG} {payload}"


def emit_bedrock_turn(agent, response):
    """Write one ``llm_usage`` line for a completed Bedrock turn. FAIL-OPEN — never raises.

    Reads the turn's token usage + model off the normalized ``response`` and the Zulip
    conversation context off the ``agent`` (``platform`` + ``_chat_*`` set in
    ``agent/agent_init.py``), builds the C1 event, and prints the tagged line to stdout for the
    tenant Vector to ship. Any error (missing attr, bad usage, anything) is swallowed: telemetry
    must never break the turn.
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            # No usage on the turn. Emit a non-tagged debug line to stderr (which reaches the
            # container docker stream; NOT the LLM_USAGE_EVENT tag, so Vector's route ignores
            # it) so a still-broken deploy is diagnosable from one live turn.
            sys.stderr.write(
                "AFAI_USAGE_DEBUG no-usage api_mode=%s provider=%s resp=%s\n"
                % (getattr(agent, "api_mode", None), getattr(agent, "provider", None),
                   type(response).__name__)
            )
            sys.stderr.flush()
            return
        # Usage attr names differ by adapter: bedrock_converse builds prompt_tokens/
        # completion_tokens; other paths may expose input_tokens/output_tokens or a dict.
        def _u(*names):
            for n in names:
                v = getattr(usage, n, None)
                if v is None and isinstance(usage, dict):
                    v = usage.get(n)
                if v is not None:
                    return v
            return 0
        tokens_in = _u("prompt_tokens", "input_tokens", "inputTokens")
        tokens_out = _u("completion_tokens", "output_tokens", "outputTokens")
        model = getattr(response, "model", None) or getattr(agent, "model", None)

        realm = os.environ.get("TENANT_NAME", "") or "unknown"
        platform = getattr(agent, "platform", None)
        chat_type = getattr(agent, "_chat_type", None)
        stream = getattr(agent, "_chat_name", None)
        chat_id = getattr(agent, "_chat_id", None)
        topic = getattr(agent, "_thread_id", None)
        # For a Zulip stream turn the topic is also carried in chat_id ("{stream_id}:{topic}");
        # recover it there if _thread_id was not populated.
        if not topic and chat_id and ":" in str(chat_id):
            topic = str(chat_id).split(":", 1)[1]

        event = build_event(
            realm=realm,
            platform=platform,
            chat_type=chat_type,
            stream=stream,
            topic=topic,
            chat_id=chat_id,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        # Raw line to STDERR (NOT stdout, NOT logging). Under s6-overlay the gateway's stdout
        # is captured to an s6 logfile and never reaches the container docker stream Vector
        # tails; the gateway's stderr DOES reach it. A bare (un-prefixed) line preserves the
        # `LLM_USAGE_EVENT ` prefix Vector's route matches. flush so it ships promptly.
        sys.stderr.write(format_event_line(event) + "\n")
        sys.stderr.flush()
    except Exception:
        # Fail-open: a telemetry failure must never propagate into the agent's turn.
        return
