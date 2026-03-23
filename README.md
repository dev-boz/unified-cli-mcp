# unified-cli-mcp

Standalone MCP server that exposes installed coding CLIs through one universal `ask` tool.

It lets one MCP client call another local CLI directly. For example:
- Claude Code calling Codex
- Codex calling Gemini
- Copilot calling Cursor

## Supported backends

Working in live smoke tests on this machine:
- `gemini`
- `codex`
- `claude`
- `kiro`
- `kilo`
- `copilot`
- `opencode`
- `mistral`
- `cursor`

Known exclusions:
- `qodo`: excluded for now because upstream execution currently returns a sunset response
- `amp`: installed, but headless execute mode currently requires paid credits

## What it exposes

Two MCP tools:
- `ask(backend, prompt, model?, cwd?, timeout?)`
- `backends()`

`ask` runs a one-shot headless CLI call.

`backends` reports which configured local binaries are available.

## Install

### Run directly from GitHub with `uvx`

```bash
uvx --from git+https://github.com/dev-boz/unified-cli-mcp unified-cli-mcp
```

### Install as a tool with `uv`

```bash
uv tool install git+https://github.com/dev-boz/unified-cli-mcp
```

### Install with `pipx`

```bash
pipx install git+https://github.com/dev-boz/unified-cli-mcp
```

### Local development

```bash
git clone https://github.com/dev-boz/unified-cli-mcp
cd unified-cli-mcp
uv sync
uv run unified-cli-mcp
```

## MCP client config

### Generic stdio config

```json
{
  "mcpServers": {
    "unified-cli-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/dev-boz/unified-cli-mcp",
        "unified-cli-mcp"
      ]
    }
  }
}
```

### Optional environment

```json
{
  "UNIFIED_CLI_ENABLED_BACKENDS": "gemini,codex,claude,copilot,cursor,mistral",
  "UNIFIED_CLI_DEFAULT_TIMEOUT": "300"
}
```

## Example prompts

Inside any MCP client:

```text
Use the unified-cli-mcp ask tool with backend="codex", cwd="/path/to/repo", prompt="Review this project for parsing bugs. Return only the top 3 findings."
```

```text
Use unified-cli-mcp to ask gemini: "Summarize the architecture of this repo in 5 bullets."
```

```text
Call the ask tool with backend="cursor", prompt="Suggest a refactor plan for the parser layer. No code."
```

## Notes

- The target CLI must already be installed and authenticated on the machine where this MCP server runs.
- This server is standalone. It does not require `cli-agent-nexus`.
- Backend aliases like `claude-code`, `mistral-vibe`, and `cursor-agent` are normalized internally.

## Development

```bash
python3 -m pytest tests/test_core.py -q
python3 -m compileall unified_cli_mcp
```
