from __future__ import annotations

import asyncio

from unified_cli_mcp.core import (
    DEFAULT_ENABLED_BACKENDS,
    SUPPORTED_BACKENDS,
    UnifiedCLICore,
    _detect_semantic_error,
    _parse_event_stream_output,
    _parse_kiro_output,
    _parse_structured_output,
    extract_text_response,
    parse_enabled_backends,
)


class FakeExecutor:
    def __init__(self, discovered: dict | None = None, responses: dict | None = None):
        self._discovered = discovered or {}
        self._responses = responses or {}
        self.calls: list[tuple[str, str | None, str, str | None, float | None]] = []

    async def discover_backends(self, backend_names):
        result = {}
        for backend_name in backend_names:
            result[backend_name] = self._discovered.get(
                backend_name,
                {
                    "available": False,
                    "binary": f"{SUPPORTED_BACKENDS[backend_name].binary} not found",
                    "default_model": SUPPORTED_BACKENDS[backend_name].default_model,
                    "description": SUPPORTED_BACKENDS[backend_name].description,
                },
            )
        return result

    async def run(self, spec, prompt: str, *, model=None, cwd=None, timeout=None):
        self.calls.append((spec.name, model, prompt, cwd, timeout))
        response = self._responses.get(spec.name, "")
        if isinstance(response, Exception):
            raise response
        return response


def test_parse_enabled_backends_filters_unknown_names_and_normalizes_aliases():
    assert parse_enabled_backends("gemini,unknown,codex,gemini") == ["gemini", "codex"]
    assert parse_enabled_backends("claude code,mistral vibe,cursor-agent") == [
        "claude",
        "mistral",
        "cursor",
    ]
    assert parse_enabled_backends("") == list(DEFAULT_ENABLED_BACKENDS)
    assert "qodo" not in DEFAULT_ENABLED_BACKENDS


def test_extract_text_response_reads_anthropic_content():
    response = {
        "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
    }
    assert extract_text_response(response) == "first\nsecond"


def test_extract_text_response_reads_assistant_message_arrays():
    response = [
        {"role": "system", "content": "ignore"},
        {"role": "assistant", "content": "OK"},
    ]
    assert extract_text_response(response) == "OK"


def test_parse_event_stream_output_handles_text_and_result_events():
    stdout = "\n".join(
        [
            '{"type":"system","subtype":"init"}',
            '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"prompt"}]}}',
            '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"OK"}]}}',
            '{"type":"result","subtype":"success","is_error":false,"result":"OK"}',
        ]
    )
    assert _parse_event_stream_output(stdout) == "OK"


def test_parse_structured_output_handles_json_message_arrays():
    stdout = '[{"role":"system","content":"ignore"},{"role":"assistant","content":"OK"}]'
    assert _parse_structured_output(stdout) == "OK"


def test_parse_kiro_output_strips_blockquote_prefix():
    assert _parse_kiro_output("> OK") == "OK"


def test_detect_semantic_error_handles_json_errors_and_qodo_sunset():
    opencode_error = (
        '{"type":"error","error":{"name":"UnknownError","data":{"message":"missing project"}}}'
    )
    amp_error = (
        '{"type":"result","subtype":"error_during_execution","is_error":true,"error":"paid credits required"}'
    )

    assert _detect_semantic_error(opencode_error, "opencode") == "UnknownError: missing project"
    assert _detect_semantic_error(amp_error, "amp") == "paid credits required"
    assert (
        _detect_semantic_error(
            "Qodo Command has been sunset and is no longer available.",
            "qodo",
        )
        == "Qodo Command has been sunset and is no longer available."
    )


def test_backend_defaults_match_known_working_models():
    assert SUPPORTED_BACKENDS["kiro"].default_model == "claude-sonnet-4.5"
    assert SUPPORTED_BACKENDS["opencode"].default_model == "opencode/big-pickle"


def test_discover_backends_filters_to_enabled_support():
    fake = FakeExecutor(
        discovered={
            "gemini": {"available": True, "binary": "/usr/bin/gemini"},
            "codex": {"available": False, "binary": "codex not found"},
            "cursor": {"available": True, "binary": "/usr/bin/agent"},
        }
    )
    core = UnifiedCLICore(executor=fake, enabled_backends=["gemini", "codex"])

    result = asyncio.run(core.discover_backends())

    assert list(result.keys()) == ["gemini", "codex"]
    assert result["gemini"]["available"] is True
    assert result["codex"]["available"] is False


def test_backends_tool_returns_structured_list():
    fake = FakeExecutor(
        discovered={
            "gemini": {"available": True, "binary": "/usr/bin/gemini"},
            "codex": {"available": True, "binary": "/usr/bin/codex"},
        }
    )
    core = UnifiedCLICore(executor=fake, enabled_backends=["gemini", "codex"])
    asyncio.run(core.discover_backends())

    result = asyncio.run(core.execute_tool("backends"))

    assert result.is_error is False
    assert result.structured_content is not None
    assert [item["name"] for item in result.structured_content["backends"]] == [
        "gemini",
        "codex",
    ]


def test_ask_uses_default_model_for_enabled_backend():
    fake = FakeExecutor(
        discovered={
            "gemini": {"available": True, "binary": "/usr/bin/gemini"},
        },
        responses={"gemini": "hello from gemini"},
    )
    core = UnifiedCLICore(executor=fake, enabled_backends=["gemini"])
    asyncio.run(core.discover_backends())

    result = asyncio.run(
        core.execute_tool(
            "ask",
            {"backend": "gemini", "prompt": "say hi"},
        )
    )

    assert result.text == "hello from gemini"
    assert fake.calls == [
        ("gemini", "gemini-3.1-pro-preview", "say hi", None, None),
    ]


def test_ask_passes_model_override_cwd_and_timeout():
    fake = FakeExecutor(
        discovered={
            "claude": {"available": True, "binary": "/usr/bin/claude"},
        },
        responses={"claude": "hello from claude"},
    )
    core = UnifiedCLICore(executor=fake, enabled_backends=["claude"])
    asyncio.run(core.discover_backends())

    result = asyncio.run(
        core.execute_tool(
            "ask",
            {
                "backend": "claude",
                "prompt": "review this",
                "model": "claude-opus-4-6",
                "cwd": "/tmp",
                "timeout": 42,
            },
        )
    )

    assert result.text == "hello from claude"
    assert fake.calls == [
        ("claude", "claude-opus-4-6", "review this", "/tmp", 42.0),
    ]


def test_ask_rejects_unavailable_backend():
    fake = FakeExecutor(
        discovered={
            "copilot": {"available": False, "binary": "copilot not found"},
        }
    )
    core = UnifiedCLICore(executor=fake, enabled_backends=["copilot"])
    asyncio.run(core.discover_backends())

    result = asyncio.run(
        core.execute_tool(
            "ask",
            {"backend": "copilot", "prompt": "test"},
        )
    )

    assert result.is_error is True
    assert "not available" in result.text


def test_ask_rejects_missing_cwd():
    fake = FakeExecutor(
        discovered={
            "gemini": {"available": True, "binary": "/usr/bin/gemini"},
        },
        responses={"gemini": "unused"},
    )
    core = UnifiedCLICore(executor=fake, enabled_backends=["gemini"])

    result = asyncio.run(
        core.execute_tool(
            "ask",
            {"backend": "gemini", "prompt": "test", "cwd": "/definitely/missing"},
        )
    )

    assert result.is_error is True
    assert "does not exist" in result.text
