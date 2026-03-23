from __future__ import annotations

import asyncio
from types import SimpleNamespace

from unified_cli_mcp.core import (
    DEFAULT_ENABLED_BACKENDS,
    LocalCLIExecutor,
    SUPPORTED_BACKENDS,
    UnifiedCLICore,
    _detect_semantic_error,
    _dangerous_flags_enabled,
    _copilot_command,
    _cursor_command,
    _claude_command,
    _codex_command,
    _kiro_command,
    _qodo_command,
    _amp_command,
    _parse_event_stream_output,
    _parse_kiro_output,
    _parse_structured_output,
    extract_text_response,
    parse_enabled_backends,
)
from unified_cli_mcp.server import _to_mcp_result


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


def test_dangerous_flags_are_opt_in(monkeypatch):
    monkeypatch.delenv("UNIFIED_CLI_ALLOW_DANGEROUS", raising=False)
    assert _dangerous_flags_enabled() is False
    assert "--full-auto" not in _codex_command("test", None, None)
    assert "--dangerously-skip-permissions" not in _claude_command("test", None, None)
    assert "--trust-all-tools" not in _kiro_command("test", None, None)
    assert "--allow-all" not in _copilot_command("test", None, None)
    assert "--approve-mcps" not in _cursor_command("test", None, None)
    assert "--dangerously-allow-all" not in _amp_command("test", None, None)
    assert "--permissions=r" in _qodo_command("test", None, None)

    monkeypatch.setenv("UNIFIED_CLI_ALLOW_DANGEROUS", "1")
    assert _dangerous_flags_enabled() is True
    assert "--full-auto" in _codex_command("test", None, None)
    assert "--dangerously-skip-permissions" in _claude_command("test", None, None)
    assert "--trust-all-tools" in _kiro_command("test", None, None)
    assert "--allow-all" in _copilot_command("test", None, None)
    assert "--approve-mcps" in _cursor_command("test", None, None)
    assert "--dangerously-allow-all" in _amp_command("test", None, None)
    assert "--permissions=rwx" in _qodo_command("test", None, None)


def test_invalid_default_timeout_falls_back_to_300(monkeypatch):
    monkeypatch.setenv("UNIFIED_CLI_DEFAULT_TIMEOUT", "not-a-number")
    executor = LocalCLIExecutor()
    assert executor.default_timeout == 300.0


def test_to_mcp_result_preserves_is_error_and_structured_content():
    result = SimpleNamespace(text="boom", is_error=True, structured_content={"x": 1})

    class FakeTextContent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCallToolResult:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    converted = _to_mcp_result(FakeCallToolResult, FakeTextContent, result)

    assert converted.kwargs["isError"] is True
    assert converted.kwargs["structuredContent"] == {"x": 1}
    assert converted.kwargs["content"][0].kwargs == {"type": "text", "text": "boom"}


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
