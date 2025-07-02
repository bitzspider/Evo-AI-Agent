#!/usr/bin/env python3
"""
Simple Time MCP Server
Provides a single tool to get current time
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Sequence
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Initialize the MCP server
app = Server("time-server")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="get_current_time",
            description="Get the current date and time",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Time format (default: human-readable)",
                        "enum": ["human", "iso", "timestamp"],
                        "default": "human"
                    }
                },
                "additionalProperties": False
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Execute tool."""
    
    if name == "get_current_time":
        format_type = arguments.get("format", "human")
        now = datetime.now()
        
        if format_type == "iso":
            time_str = now.isoformat()
        elif format_type == "timestamp":
            time_str = str(now.timestamp())
        else:  # human
            time_str = now.strftime("%A, %B %d, %Y at %I:%M:%S %p")
        
        return [
            types.TextContent(
                type="text",
                text=f"Current time: {time_str}"
            )
        ]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main server entry point."""
    async with stdio_server() as streams:
        await app.run(
            streams[0], 
            streams[1], 
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main()) 