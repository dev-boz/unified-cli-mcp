#!/usr/bin/env python3
"""Unified CLI MCP Server.

Exposes all cli-agent-nexus backends as MCP tools for lightweight queries.
Dynamically discovers backends and registers tools on startup.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from nexus_client import NexusClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("unified-cli-mcp")

# Default model mappings for each backend
DEFAULT_MODELS = {
    "gemini": "gemini-3-pro-preview",
    "kiro": "claude-sonnet-4-6",
    "codex": "codex-gpt-5.3",
    "cursor": "cursor-auto",
    "qodo": "qodo-gpt-5.2",
    "amp": "amp-default",
    "opencode": "opencode-default",
    "kilo": "kilo-default",
    "mistral": "mistral-default",
    "copilot": "copilot-gpt-5.2",
    "copilot-acp": "copilot-acp-gpt-5.2",
}

# Backend descriptions for tool documentation
BACKEND_DESCRIPTIONS = {
    "gemini": "Google Gemini 3.1 Pro (via gemini CLI with OAuth)",
    "kiro": "Claude Sonnet 4.6 (via Kiro ACP with free tier)",
    "codex": "OpenAI GPT-5.3 Codex (via codex CLI)",
    "cursor": "Cursor Agent (auto model selection)",
    "qodo": "Qodo multi-model access (GPT-5.x, Claude 4.x, O-series)",
    "amp": "Amp deep reasoning mode (Claude Opus 4.6)",
    "opencode": "OpenCode multi-model router (1200+ models)",
    "kilo": "Kilo free model access (30+ models)",
    "mistral": "Mistral Vibe (devstral, magistral models)",
    "copilot": "GitHub Copilot (GPT-5.x models)",
    "copilot-acp": "GitHub Copilot ACP (17 models)",
}


class UnifiedCLIMCP:
    """MCP server that exposes cli-agent-nexus backends as tools."""

    def __init__(self):
        self.server = Server("unified-cli-mcp")
        self.nexus = NexusClient()
        self.available_backends: dict[str, dict[str, Any]] = {}
        self._setup_handlers()

    def _setup_handlers(self):
        """Register MCP protocol handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List all available backend tools."""
            tools = []
            for backend_name, backend_info in self.available_backends.items():
                if not backend_info.get("available", False):
                    continue

                default_model = DEFAULT_MODELS.get(backend_name, f"{backend_name}-default")
                description = BACKEND_DESCRIPTIONS.get(
                    backend_name,
                    f"Query {backend_name} backend via cli-agent-nexus"
                )

                tools.append(Tool(
                    name=f"ask-{backend_name}",
                    description=description,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt/question to send to the model",
                            },
                            "model": {
                                "type": "string",
                                "description": f"Optional model override (default: {default_model})",
                            },
                        },
                        "required": ["prompt"],
                    },
                ))

            logger.info("Registered %d tools for available backends", len(tools))
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool invocation."""
            # Extract backend name from tool name (ask-{backend})
            if not name.startswith("ask-"):
                return [TextContent(
                    type="text",
                    text=f"Error: Unknown tool '{name}'"
                )]

            backend_name = name[4:]  # Remove "ask-" prefix

            # Validate backend is available
            if backend_name not in self.available_backends:
                return [TextContent(
                    type="text",
                    text=f"Error: Backend '{backend_name}' not found"
                )]

            backend_info = self.available_backends[backend_name]
            if not backend_info.get("available", False):
                return [TextContent(
                    type="text",
                    text=f"Error: Backend '{backend_name}' is not available"
                )]

            # Extract arguments
            prompt = arguments.get("prompt")
            if not prompt:
                return [TextContent(
                    type="text",
                    text="Error: Missing required argument 'prompt'"
                )]

            # Determine model to use
            model = arguments.get("model")
            if not model:
                model = DEFAULT_MODELS.get(backend_name, f"{backend_name}-default")

            logger.info("Calling backend=%s model=%s prompt_len=%d",
                       backend_name, model, len(prompt))

            # Send message to nexus
            try:
                response = await self.nexus.send_message(model, prompt)

                # Extract text from response
                # Anthropic Messages API format: {"content": [{"type": "text", "text": "..."}]}
                if isinstance(response, dict):
                    content = response.get("content", [])
                    if content and isinstance(content, list):
                        text_parts = [
                            block.get("text", "")
                            for block in content
                            if block.get("type") == "text"
                        ]
                        result_text = "\n".join(text_parts)
                    else:
                        # Fallback: stringify the whole response
                        result_text = str(response)
                else:
                    result_text = str(response)

                logger.info("Backend %s returned %d chars", backend_name, len(result_text))
                return [TextContent(type="text", text=result_text)]

            except TimeoutError as e:
                logger.error("Timeout calling backend %s: %s", backend_name, e)
                return [TextContent(
                    type="text",
                    text=f"Error: Request timed out - {e}"
                )]
            except ConnectionError as e:
                logger.error("Connection error calling backend %s: %s", backend_name, e)
                return [TextContent(
                    type="text",
                    text=f"Error: Cannot connect to nexus - {e}"
                )]
            except RuntimeError as e:
                logger.error("Runtime error calling backend %s: %s", backend_name, e)
                return [TextContent(
                    type="text",
                    text=f"Error: {e}"
                )]
            except Exception as e:
                logger.exception("Unexpected error calling backend %s", backend_name)
                return [TextContent(
                    type="text",
                    text=f"Error: Unexpected error - {e}"
                )]

    async def discover_backends(self):
        """Query nexus to discover available backends."""
        logger.info("Discovering backends from nexus at %s", self.nexus.base_url)

        try:
            result = await self.nexus.get_backends()
            self.available_backends = result.get("backends", {})

            available_count = sum(
                1 for b in self.available_backends.values()
                if b.get("available", False)
            )

            logger.info(
                "Discovered %d backends (%d available): %s",
                len(self.available_backends),
                available_count,
                ", ".join(self.available_backends.keys()),
            )

        except Exception as e:
            logger.error("Failed to discover backends: %s", e)
            logger.warning("Starting with empty backend list")
            self.available_backends = {}

    async def run(self):
        """Run the MCP server."""
        # Discover backends before starting server
        await self.discover_backends()

        if not self.available_backends:
            logger.warning("No backends discovered - server will have no tools")

        # Run the server
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Unified CLI MCP server started")
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


def main():
    """Entry point."""
    try:
        server = UnifiedCLIMCP()
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception:
        logger.exception("Server crashed")
        sys.exit(1)


if __name__ == "__main__":
    main()
