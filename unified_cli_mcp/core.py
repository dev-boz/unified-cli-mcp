"""Core logic for the standalone unified CLI MCP server.

This module is intentionally independent of the ``mcp`` SDK so backend
discovery, local CLI execution, and response handling can be tested without the
runtime dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger("unified-cli-mcp.core")

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
MISTRAL_AGENT_MODES = {"default", "plan", "accept-edits", "auto-approve", "chat"}
QODO_SUNSET_TEXT = "qodo command has been sunset and is no longer available"


def _clean_text(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text).strip()


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _dangerous_flags_enabled() -> bool:
    return _env_flag_enabled("UNIFIED_CLI_ALLOW_DANGEROUS")


def _stringify_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def extract_text_response(response: dict[str, Any] | str) -> str:
    """Extract assistant text from structured or plain responses."""
    if isinstance(response, str):
        return _clean_text(response)
    if isinstance(response, list):
        assistant_parts: list[str] = []
        parts: list[str] = []
        for item in response:
            text = extract_text_response(item)
            if not text:
                continue
            if isinstance(item, dict) and item.get("role") == "assistant":
                assistant_parts.append(text)
            else:
                parts.append(text)
        if assistant_parts:
            return assistant_parts[-1]
        if parts:
            return "\n".join(parts).strip()
        return _stringify_json(response)

    update = response.get("params", {}).get("update", {})
    if update.get("sessionUpdate") == "agent_message_chunk":
        content = update.get("content", {})
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str) and text:
                return text

    item = response.get("item")
    if isinstance(item, dict) and item.get("type") == "agent_message":
        text = item.get("text")
        if isinstance(text, str) and text:
            return text

    part = response.get("part")
    if isinstance(part, dict):
        text = part.get("text")
        if isinstance(text, str) and text:
            return text

    message = response.get("message")
    if isinstance(message, dict):
        text = _extract_text_from_content(message.get("content"))
        if text:
            return text

    content = response.get("content")
    text = _extract_text_from_content(content)
    if text:
        return text

    role = response.get("role")
    if role == "assistant" and isinstance(content, str) and content:
        return content

    if response.get("type") == "message" and role == "assistant" and isinstance(content, str):
        return content

    if response.get("type") == "result" and response.get("is_error") is False:
        result = response.get("result")
        if isinstance(result, str) and result:
            return result

    return _stringify_json(response)


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _parse_json_lines_output(stdout: str) -> str:
    parts: list[str] = []
    fallback_lines: list[str] = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            fallback_lines.append(_clean_text(raw_line))
            continue

        text = extract_text_response(parsed)
        if text and not text.startswith("{"):
            parts.append(text)

    if parts:
        return "".join(parts).strip()
    if fallback_lines:
        return "\n".join(line for line in fallback_lines if line).strip()
    return _clean_text(stdout)


def _parse_event_stream_output(stdout: str) -> str:
    parts: list[str] = []
    final_result: str | None = None
    fallback_lines: list[str] = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            fallback_lines.append(_clean_text(raw_line))
            continue

        if not isinstance(parsed, dict):
            text = extract_text_response(parsed)
            if text and text != _stringify_json(parsed):
                parts.append(text)
            continue

        event_type = parsed.get("type")
        if event_type == "text":
            part = parsed.get("part", {})
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
                    continue

        if event_type == "assistant":
            message = parsed.get("message", {})
            if isinstance(message, dict) and message.get("role") == "assistant":
                text = _extract_text_from_content(message.get("content"))
                if text:
                    parts.append(text)
                    continue

        if event_type == "result" and parsed.get("is_error") is False:
            result = parsed.get("result")
            if isinstance(result, str) and result.strip():
                final_result = result.strip()
                continue

        if parsed.get("role") == "assistant":
            text = extract_text_response(parsed)
            if text and text != _stringify_json(parsed):
                parts.append(text)

    if final_result:
        return _clean_text(final_result)
    if parts:
        return "".join(parts).strip()
    if fallback_lines:
        return "\n".join(line for line in fallback_lines if line).strip()
    return _clean_text(stdout)


def _parse_structured_output(stdout: str) -> str:
    cleaned = _clean_text(stdout)
    if not cleaned:
        return ""

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return _parse_event_stream_output(stdout)

    text = extract_text_response(parsed)
    if text and text != _stringify_json(parsed):
        return _clean_text(text)
    return cleaned


def _parse_codex_output(stdout: str) -> str:
    completed_messages: list[str] = []
    delta_messages: list[str] = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue

        if parsed.get("type") == "item.completed":
            item = parsed.get("item", {})
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text:
                    completed_messages.append(text)
        elif parsed.get("type") == "item.delta":
            item = parsed.get("item", {})
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text:
                    delta_messages.append(text)

    if completed_messages:
        return completed_messages[-1].strip()
    if delta_messages:
        return "".join(delta_messages).strip()
    return _parse_json_lines_output(stdout)


def _parse_plain_output(stdout: str) -> str:
    return _clean_text(stdout)


def _parse_kiro_output(stdout: str) -> str:
    cleaned = _clean_text(stdout)
    if not cleaned:
        return ""

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines and all(line.startswith(">") for line in lines):
        stripped = [line[1:].lstrip() for line in lines]
        return "\n".join(stripped).strip()
    return cleaned


def _error_payload_to_text(error: Any) -> str | None:
    if isinstance(error, str):
        return _clean_text(error)
    if not isinstance(error, dict):
        return None

    message = error.get("message")
    if not isinstance(message, str):
        data = error.get("data")
        if isinstance(data, dict):
            nested = data.get("message")
            if isinstance(nested, str):
                message = nested

    name = error.get("name")
    name_text = name if isinstance(name, str) else None

    if name_text and message and name_text not in message:
        return f"{name_text}: {message}"
    if message:
        return _clean_text(message)
    if name_text:
        return _clean_text(name_text)
    return None


def _extract_semantic_error(response: Any) -> str | None:
    if isinstance(response, list):
        for item in response:
            error = _extract_semantic_error(item)
            if error:
                return error
        return None

    if not isinstance(response, dict):
        return None

    if response.get("type") == "error":
        return _error_payload_to_text(response.get("error")) or _stringify_json(response)

    subtype = response.get("subtype")
    if response.get("is_error") is True or (
        isinstance(subtype, str) and subtype.startswith("error")
    ):
        error = _error_payload_to_text(response.get("error"))
        if error:
            return error
        result = response.get("result")
        if isinstance(result, str) and result:
            return _clean_text(result)
        return _stringify_json(response)

    error = response.get("error")
    if error:
        return _error_payload_to_text(error)

    return None


def _detect_semantic_error(stdout: str, backend_name: str) -> str | None:
    cleaned = _clean_text(stdout)
    if not cleaned:
        return None

    if backend_name == "qodo" and QODO_SUNSET_TEXT in cleaned.lower():
        return cleaned

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed_line = json.loads(line)
            except json.JSONDecodeError:
                continue
            error = _extract_semantic_error(parsed_line)
            if error:
                return error
        return None

    return _extract_semantic_error(parsed)


CommandBuilder = Callable[[str, str | None, str | None], list[str]]
OutputParser = Callable[[str], str]


def _gemini_command(prompt: str, model: str | None, cwd: str | None) -> list[str]:
    cmd = ["gemini", "-p", prompt, "-y", "-o", "text"]
    if model:
        cmd.extend(["-m", model])
    return cmd


def _codex_command(prompt: str, model: str | None, cwd: str | None) -> list[str]:
    cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--json"]
    if cwd:
        cmd.extend(["-C", cwd])
    if _dangerous_flags_enabled():
        cmd.append("--full-auto")
    if model:
        cmd.extend(["-m", model])
    cmd.append(prompt)
    return cmd


def _claude_command(prompt: str, model: str | None, cwd: str | None) -> list[str]:
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "text",
        "--no-session-persistence",
    ]
    if _dangerous_flags_enabled():
        cmd.append("--dangerously-skip-permissions")
    if cwd:
        cmd.extend(["--add-dir", cwd])
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)
    return cmd


def _kiro_command(prompt: str, model: str | None, cwd: str | None) -> list[str]:
    cmd = ["kiro-cli", "chat", prompt, "--no-interactive"]
    if _dangerous_flags_enabled():
        cmd.append("--trust-all-tools")
    if model:
        cmd.extend(["--model", model])
    return cmd


def _kilo_command(prompt: str, model: str | None, cwd: str | None) -> list[str]:
    cmd = ["kilo", "run", "--auto", "--format", "json"]
    if cwd:
        cmd.extend(["--dir", cwd])
    if model:
        cmd.extend(["-m", model])
    cmd.append(prompt)
    return cmd


def _copilot_command(prompt: str, model: str | None, cwd: str | None) -> list[str]:
    cmd = [
        "copilot",
        "-p",
        prompt,
        "-s",
        "--no-color",
        "--reasoning-effort",
        "high",
    ]
    if _dangerous_flags_enabled():
        cmd.append("--allow-all")
    if cwd:
        cmd.extend(["--add-dir", cwd])
    if model:
        cmd.extend(["--model", model])
    return cmd


def _opencode_command(prompt: str, model: str | None, cwd: str | None) -> list[str]:
    cmd = ["opencode", "run", "--format", "json"]
    if cwd:
        cmd.extend(["--dir", cwd])
    if model:
        cmd.extend(["-m", model])
    cmd.append(prompt)
    return cmd


def _mistral_command(prompt: str, model: str | None, cwd: str | None) -> list[str]:
    cmd = ["vibe", "--prompt", prompt, "--output", "json", "--max-turns", "10"]
    if cwd:
        cmd.extend(["--workdir", cwd])
    agent = model if model in MISTRAL_AGENT_MODES else "auto-approve"
    cmd.extend(["--agent", agent])
    return cmd


def _cursor_command(prompt: str, model: str | None, cwd: str | None) -> list[str]:
    cmd = [
        "agent",
        "-p",
        "--output-format",
        "stream-json",
        "--force",
    ]
    if _dangerous_flags_enabled():
        cmd.extend(["--approve-mcps", "--trust"])
    if cwd:
        cmd.extend(["--workspace", cwd])
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)
    return cmd


def _qodo_command(prompt: str, model: str | None, cwd: str | None) -> list[str]:
    cmd = [
        "qodo",
        "--ci",
        "--yes",
        "--silent",
        "--permissions=rwx" if _dangerous_flags_enabled() else "--permissions=r",
        "--tools=shell,git,filesystem",
    ]
    if cwd:
        cmd.extend(["--dir", cwd])
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)
    return cmd


def _amp_command(prompt: str, model: str | None, cwd: str | None) -> list[str]:
    cmd = ["amp", "-x", prompt, "--stream-json", "--no-color"]
    if _dangerous_flags_enabled():
        cmd.append("--dangerously-allow-all")
    if model:
        cmd.extend(["--mode", model])
    return cmd


def _qwen_command(prompt: str, model: str | None, cwd: str | None) -> list[str]:
    cmd = ["qwen", "-p", prompt, "--output-format", "text"]
    if _dangerous_flags_enabled():
        cmd.append("--yolo")
    if cwd:
        cmd.extend(["--include-directories", cwd])
    if model:
        cmd.extend(["-m", model])
    return cmd


@dataclass(frozen=True)
class BackendSpec:
    """Static metadata for a supported local CLI backend."""

    name: str
    binary: str
    default_model: str | None
    description: str
    build_command: CommandBuilder = field(repr=False)
    output_parser: OutputParser = field(repr=False, default=_parse_plain_output)
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    """Portable tool definition used by the MCP adapter."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Portable tool result used by the MCP adapter."""

    text: str
    is_error: bool = False
    structured_content: dict[str, Any] | None = None


SUPPORTED_BACKENDS: dict[str, BackendSpec] = {
    "gemini": BackendSpec(
        name="gemini",
        binary="gemini",
        default_model="gemini-3.1-pro-preview",
        description="Ask Gemini directly via the Gemini CLI.",
        build_command=_gemini_command,
        output_parser=_parse_plain_output,
        aliases=("gemini-cli",),
    ),
    "codex": BackendSpec(
        name="codex",
        binary="codex",
        default_model="gpt-5.4",
        description="Ask Codex directly via the Codex CLI.",
        build_command=_codex_command,
        output_parser=_parse_codex_output,
        aliases=("codex-cli",),
    ),
    "claude": BackendSpec(
        name="claude",
        binary="claude",
        default_model="claude-sonnet-4-6",
        description="Ask Claude Code directly via the Claude CLI.",
        build_command=_claude_command,
        output_parser=_parse_plain_output,
        aliases=("claude-code", "claude-code-cli", "claude-cli"),
    ),
    "kiro": BackendSpec(
        name="kiro",
        binary="kiro-cli",
        default_model="claude-sonnet-4.5",
        description="Ask Kiro directly via the Kiro CLI.",
        build_command=_kiro_command,
        output_parser=_parse_kiro_output,
        aliases=("kiro-cli",),
    ),
    "kilo": BackendSpec(
        name="kilo",
        binary="kilo",
        default_model=None,
        description="Ask Kilo directly via the Kilo CLI.",
        build_command=_kilo_command,
        output_parser=_parse_event_stream_output,
        aliases=("kilo-cli",),
    ),
    "copilot": BackendSpec(
        name="copilot",
        binary="copilot",
        default_model="gpt-5-mini",
        description="Ask GitHub Copilot directly via the Copilot CLI.",
        build_command=_copilot_command,
        output_parser=_parse_plain_output,
        aliases=("copilot-cli", "github-copilot"),
    ),
    "opencode": BackendSpec(
        name="opencode",
        binary="opencode",
        default_model="opencode/big-pickle",
        description="Ask OpenCode directly via the OpenCode CLI.",
        build_command=_opencode_command,
        output_parser=_parse_event_stream_output,
        aliases=("opencode-cli",),
    ),
    "mistral": BackendSpec(
        name="mistral",
        binary="vibe",
        default_model="devstral-2",
        description="Ask Mistral Vibe directly via the Vibe CLI.",
        build_command=_mistral_command,
        output_parser=_parse_structured_output,
        aliases=("mistral-vibe", "mistral-vibe-cli", "vibe", "vibe-cli"),
    ),
    "cursor": BackendSpec(
        name="cursor",
        binary="agent",
        default_model="auto",
        description="Ask Cursor Agent directly via the Cursor CLI.",
        build_command=_cursor_command,
        output_parser=_parse_event_stream_output,
        aliases=("cursor-cli", "cursor-agent"),
    ),
    "qodo": BackendSpec(
        name="qodo",
        binary="qodo",
        default_model=None,
        description="Ask Qodo directly via the Qodo CLI.",
        build_command=_qodo_command,
        output_parser=_parse_plain_output,
        aliases=("qodo-cli",),
    ),
    "amp": BackendSpec(
        name="amp",
        binary="amp",
        default_model=None,
        description="Ask Amp directly via the Amp CLI.",
        build_command=_amp_command,
        output_parser=_parse_event_stream_output,
        aliases=("amp-cli",),
    ),
    "qwen": BackendSpec(
        name="qwen",
        binary="qwen",
        default_model="qwen3-coder-plus",
        description="Ask Qwen Code directly via the Qwen CLI.",
        build_command=_qwen_command,
        output_parser=_parse_plain_output,
        aliases=("qwen-code", "qwen-cli", "qwen-code-cli"),
    ),
}

DEFAULT_ENABLED_BACKENDS = (
    "gemini",
    "codex",
    "claude",
    "kiro",
    "kilo",
    "copilot",
    "opencode",
    "mistral",
    "cursor",
    "amp",
    "qwen",
)

BACKEND_ALIASES: dict[str, str] = {}
for backend_name, spec in SUPPORTED_BACKENDS.items():
    alias_values = (backend_name, *spec.aliases)
    for alias in alias_values:
        normalized = alias.replace("_", "-").replace(" ", "-").strip().lower()
        BACKEND_ALIASES[normalized] = backend_name


def normalize_backend_name(raw_name: str) -> str | None:
    normalized = raw_name.strip().lower().replace("_", "-")
    normalized = re.sub(r"\s+", "-", normalized)
    return BACKEND_ALIASES.get(normalized)


def parse_enabled_backends(raw: str | None) -> list[str]:
    """Parse a comma-separated backend list, preserving supported names only."""
    if not raw:
        return list(DEFAULT_ENABLED_BACKENDS)

    enabled: list[str] = []
    for item in raw.split(","):
        name = normalize_backend_name(item)
        if not name or name in enabled:
            continue
        enabled.append(name)

    return enabled or list(DEFAULT_ENABLED_BACKENDS)


class LocalCLIExecutor:
    """Detect and execute local CLI backends."""

    def __init__(
        self,
        which_fn: Callable[[str], str | None] | None = None,
        default_timeout: float | None = None,
    ):
        self.which_fn = which_fn or shutil.which
        if default_timeout is not None:
            self.default_timeout = default_timeout
            return

        try:
            self.default_timeout = float(os.getenv("UNIFIED_CLI_DEFAULT_TIMEOUT", "300"))
        except ValueError:
            logger.warning(
                "Invalid UNIFIED_CLI_DEFAULT_TIMEOUT=%r, falling back to 300s",
                os.getenv("UNIFIED_CLI_DEFAULT_TIMEOUT"),
            )
            self.default_timeout = 300.0

    async def discover_backends(
        self,
        backend_names: Iterable[str],
    ) -> dict[str, dict[str, Any]]:
        availability: dict[str, dict[str, Any]] = {}
        for backend_name in backend_names:
            spec = SUPPORTED_BACKENDS[backend_name]
            binary_path = self.which_fn(spec.binary)
            availability[backend_name] = {
                "available": binary_path is not None,
                "binary": binary_path or f"{spec.binary} not found",
                "default_model": spec.default_model,
                "description": spec.description,
            }
        return availability

    async def run(
        self,
        spec: BackendSpec,
        prompt: str,
        *,
        model: str | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> str:
        command = spec.build_command(prompt, model, cwd)
        effective_timeout = timeout or self.default_timeout
        logger.info(
            "Running backend=%s argv0=%s cwd=%s prompt_len=%d",
            spec.name,
            command[0],
            cwd or os.getcwd(),
            len(prompt),
        )
        env = None
        temp_dir: tempfile.TemporaryDirectory[str] | None = None

        if spec.name == "mistral":
            env, temp_dir = self._build_mistral_env(model)

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or None,
                env=env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"CLI binary '{spec.binary}' not found") from exc

        try:
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError as exc:
                process.kill()
                await process.communicate()
                raise TimeoutError(
                    f"{spec.name} timed out after {effective_timeout}s"
                ) from exc

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            if process.returncode != 0:
                error_text = _clean_text(stderr) or _clean_text(stdout)
                if not error_text:
                    error_text = f"exit code {process.returncode}"
                raise RuntimeError(f"{spec.name} failed: {error_text[:1000]}")

            semantic_error = _detect_semantic_error(stdout, spec.name)
            if semantic_error:
                raise RuntimeError(f"{spec.name} failed: {semantic_error[:1000]}")

            parsed = spec.output_parser(stdout).strip()

            if parsed:
                return parsed

            fallback = _clean_text(stdout) or _clean_text(stderr)
            if fallback:
                return fallback

            return ""
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    def _build_mistral_env(
        self,
        model: str | None,
    ) -> tuple[dict[str, str] | None, tempfile.TemporaryDirectory[str] | None]:
        if not model or model in MISTRAL_AGENT_MODES:
            return None, None

        source_home = Path.home() / ".vibe"
        temp_dir = tempfile.TemporaryDirectory(prefix="unified-cli-mcp-vibe-")
        temp_home = Path(temp_dir.name)

        if source_home.exists():
            shutil.copytree(source_home, temp_home, dirs_exist_ok=True)
        else:
            temp_home.mkdir(parents=True, exist_ok=True)

        config_path = temp_home / "config.toml"
        config_text = config_path.read_text() if config_path.exists() else ""
        quoted_model = json.dumps(model)

        if re.search(r"(?m)^\s*active_model\s*=", config_text):
            config_text = re.sub(
                r"(?m)^\s*active_model\s*=.*$",
                f"active_model = {quoted_model}",
                config_text,
                count=1,
            )
        else:
            if config_text and not config_text.endswith("\n"):
                config_text += "\n"
            config_text += f"active_model = {quoted_model}\n"

        config_path.write_text(config_text)

        env = os.environ.copy()
        env["VIBE_HOME"] = str(temp_home)
        logger.info("Using isolated VIBE_HOME=%s with active_model=%s", temp_home, model)
        return env, temp_dir


class UnifiedCLICore:
    """Core runtime for backend discovery and tool execution."""

    def __init__(
        self,
        executor: LocalCLIExecutor | Any | None = None,
        enabled_backends: Iterable[str] | None = None,
    ):
        self.executor = executor or LocalCLIExecutor()
        self.enabled_backends = list(
            enabled_backends
            or parse_enabled_backends(os.getenv("UNIFIED_CLI_ENABLED_BACKENDS"))
        )
        self.available_backends: dict[str, dict[str, Any]] = {}

    def list_tool_specs(self) -> list[ToolSpec]:
        """Return the current MCP tool definitions."""
        ask_description = (
            "Ask one configured local CLI backend a question. "
            f"Available backends: {', '.join(self.enabled_backends)}."
        )
        return [
            ToolSpec(
                name="ask",
                description=ask_description,
                input_schema={
                    "type": "object",
                    "properties": {
                        "backend": {
                            "type": "string",
                            "enum": self.enabled_backends,
                            "description": "Which backend to use.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The prompt or task to send.",
                        },
                        "model": {
                            "type": "string",
                            "description": "Optional backend-specific model override.",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Optional working directory for the CLI call.",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Optional timeout override in seconds.",
                        },
                    },
                    "required": ["backend", "prompt"],
                },
            ),
            ToolSpec(
                name="backends",
                description="List configured backends and whether their local CLI binaries are currently available.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
        ]

    async def discover_backends(self) -> dict[str, dict[str, Any]]:
        """Refresh backend availability from local CLI discovery."""
        logger.info("Discovering local CLI backends")
        self.available_backends = await self.executor.discover_backends(
            self.enabled_backends
        )
        logger.info(
            "Enabled backends: %s",
            ", ".join(
                f"{name}={'up' if info['available'] else 'down'}"
                for name, info in self.available_backends.items()
            ),
        )
        return self.available_backends

    async def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Execute a logical tool call."""
        arguments = arguments or {}

        if name == "backends":
            if not self.available_backends:
                await self.discover_backends()
            payload = self._backends_payload()
            return ToolResult(
                text=json.dumps(payload, indent=2, ensure_ascii=True),
                structured_content=payload,
            )

        if name != "ask":
            return ToolResult(text=f"Unknown tool '{name}'", is_error=True)

        backend = arguments.get("backend")
        prompt = arguments.get("prompt")
        model = arguments.get("model")
        cwd = arguments.get("cwd")
        timeout = arguments.get("timeout")

        if not backend or not isinstance(backend, str):
            return ToolResult(text="Missing required argument 'backend'", is_error=True)
        if not prompt or not isinstance(prompt, str):
            return ToolResult(text="Missing required argument 'prompt'", is_error=True)
        if backend not in self.enabled_backends:
            return ToolResult(
                text=f"Unsupported backend '{backend}'. Allowed: {', '.join(self.enabled_backends)}",
                is_error=True,
            )
        if cwd is not None and not isinstance(cwd, str):
            return ToolResult(text="Argument 'cwd' must be a string", is_error=True)
        if isinstance(cwd, str) and cwd and not os.path.isdir(cwd):
            return ToolResult(text=f"Working directory does not exist: {cwd}", is_error=True)
        if timeout is not None and not isinstance(timeout, (int, float)):
            return ToolResult(text="Argument 'timeout' must be a number", is_error=True)

        if not self.available_backends:
            await self.discover_backends()

        backend_info = self.available_backends.get(backend, {})
        if not backend_info.get("available", False):
            return ToolResult(text=f"Backend '{backend}' is not available", is_error=True)

        spec = SUPPORTED_BACKENDS[backend]
        effective_model = model if isinstance(model, str) and model else spec.default_model
        logger.info(
            "Calling backend=%s model=%s prompt_len=%d cwd=%s",
            backend,
            effective_model or "<cli-default>",
            len(prompt),
            cwd or os.getcwd(),
        )

        try:
            response_text = await self.executor.run(
                spec,
                prompt,
                model=effective_model,
                cwd=cwd if isinstance(cwd, str) and cwd else None,
                timeout=float(timeout) if timeout is not None else None,
            )
        except TimeoutError as exc:
            return ToolResult(text=str(exc), is_error=True)
        except RuntimeError as exc:
            return ToolResult(text=str(exc), is_error=True)
        except Exception as exc:
            logger.exception("Unexpected error calling backend %s", backend)
            return ToolResult(text=f"Unexpected error: {exc}", is_error=True)

        return ToolResult(text=response_text)

    def _backends_payload(self) -> dict[str, Any]:
        items = []
        for backend_name in self.enabled_backends:
            spec = SUPPORTED_BACKENDS[backend_name]
            info = self.available_backends.get(backend_name, {})
            items.append(
                {
                    "name": backend_name,
                    "available": bool(info.get("available", False)),
                    "default_model": spec.default_model,
                    "description": spec.description,
                    "binary": info.get("binary"),
                }
            )
        return {"backends": items}
