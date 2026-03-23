#!/usr/bin/env python3
"""Unified CLI MCP Server entrypoint.

The business logic lives in ``core.py`` so backend discovery and local CLI
execution can be tested without the ``mcp`` package.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from unified_cli_mcp.core import ToolSpec, UnifiedCLICore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("unified-cli-mcp")


def _import_mcp_runtime():
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except ModuleNotFoundError as exc:
        if exc.name == "mcp":
            raise RuntimeError(
                "Missing Python dependency 'mcp'. Run `uv sync` in "
                "/home/dinkum/projects/unified-cli-mcp before starting this server."
            ) from exc
        raise

    return Server, stdio_server, TextContent, Tool


def _to_mcp_tool(tool_cls, spec: ToolSpec):
    return tool_cls(
        name=spec.name,
        description=spec.description,
        inputSchema=spec.input_schema,
    )


def create_mcp_server(core: UnifiedCLICore):
    """Build the MCP SDK server around the core runtime."""
    server_cls, stdio_server, text_content_cls, tool_cls = _import_mcp_runtime()

    server = server_cls(
        "unified-cli-mcp",
        version=__import__("unified_cli_mcp").__version__,
        instructions=(
            "Use `ask` for quick synchronous queries through local CLI backends. "
            "Use `backends` to inspect current availability."
        ),
    )

    @server.list_tools()
    async def list_tools() -> list:
        return [_to_mcp_tool(tool_cls, spec) for spec in core.list_tool_specs()]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list:
        result = await core.execute_tool(name, arguments)
        return [text_content_cls(type="text", text=result.text)]

    return server, stdio_server


async def run() -> None:
    core = UnifiedCLICore()

    try:
        await core.discover_backends()
    except Exception as exc:
        logger.warning("Initial backend discovery failed: %s", exc)

    server, stdio_server = create_mcp_server(core)

    async with stdio_server() as (read_stream, write_stream):
        logger.info("Unified CLI MCP server started")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        raise SystemExit(0)
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    except Exception:
        logger.exception("Server crashed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
