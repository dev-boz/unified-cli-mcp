# Unified CLI MCP

Date: 2026-03-23

## Overview

Standalone MCP server that exposes local coding CLIs as one universal tool for Claude Code or any other MCP client.

Current backends:
`gemini`, `codex`, `claude`, `kiro`, `kilo`, `copilot`, `opencode`, `mistral`, `cursor`, `amp`

Tools:
- `ask(backend, prompt, model?, cwd?, timeout?)` — run one local CLI in headless mode
- `backends()` — list configured backends and whether the local binary is available

## Architecture

```text
Claude Code ──MCP stdio──▶ unified-cli-mcp ──subprocess──▶ local CLI binary
                                                    │
                                                    ├── gemini
                                                    ├── codex
                                                    ├── claude
                                                    ├── cursor
                                                    └── ...
```

Design: one universal `ask` tool keeps context footprint low and avoids one tool per backend.

Core logic is in `core.py`, independent of the `mcp` SDK, so it can be tested without the runtime.

## Package Structure

```text
unified-cli-mcp/
├── pyproject.toml
├── unified_cli_mcp/
│   ├── __init__.py
│   ├── __main__.py
│   ├── server.py
│   └── core.py
├── tests/
│   └── test_core.py
└── archive/
    └── server-dynamic-per-backend-2026-03-06.py
```

## Running

Prerequisites:
```bash
cd /home/dinkum/projects/unified-cli-mcp
uv sync
```

Launch methods:
```bash
uv run unified-cli-mcp
uv run python -m unified_cli_mcp
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `UNIFIED_CLI_ENABLED_BACKENDS` | all supported backends | Comma-separated backend list |
| `UNIFIED_CLI_DEFAULT_TIMEOUT` | `300` | Default timeout per CLI call in seconds |

Backend names accept common aliases such as `claude-code`, `mistral-vibe`, and `cursor-agent`.

## Adding a Backend

1. Add a `BackendSpec` entry to `SUPPORTED_BACKENDS` in `unified_cli_mcp/core.py`
2. Add aliases if the CLI is commonly referred to by another name
3. Implement a headless command builder
4. Pick a parser:
   - plain text
   - generic JSON/JSONL parser
   - backend-specific parser if needed
5. Add tests in `tests/test_core.py`

## Tests

```bash
cd /home/dinkum/projects/unified-cli-mcp
python3 -m pytest tests/test_core.py -q
python3 -m compileall unified_cli_mcp
```

## Status

- Standalone local-CLI execution is implemented
- `ask` supports `model`, `cwd`, and `timeout` overrides
- Copilot defaults to `gpt-5-mini` with high reasoning effort
- Mistral defaults to `devstral-2` via an isolated `VIBE_HOME` config override
- Qodo is intentionally excluded from the default enabled set for now because live execution still returns the upstream sunset response
- Unit tests cover alias parsing, discovery, structured `backends()`, and execution dispatch
- The server no longer requires `cli-agent-nexus`

## Next Steps

1. Add integration tests against the CLIs actually installed on this machine
2. Harden backend-specific output parsing where plain text is still noisy
3. Decide whether any backends should move from one-shot headless commands to persistent session reuse
